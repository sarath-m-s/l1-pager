"""
Phase 3: LangGraph graph integration tests.

These tests build actual StateGraph instances and exercise:
  - create_l1_pager_node: node function plugged into a hand-built graph
  - create_l1_pager_react_agent: full prebuilt ReAct loop with real routing
  - should_continue: router filtering out fetch_evicted_page calls
  - _ModelShim.bind_tools: user tools + fetch_evicted_page both present
  - wrap_model_call / WrapModelCallMiddleware
  - L1PagerGraphState: turn counter, eviction log, token totals
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from l1_pager import EvictionConfig, L1Pager, build_heap
from l1_pager.langgraph_integration import (
    L1PagerGraphState,
    create_l1_pager_node,
    create_l1_pager_react_agent,
    should_continue,
)
from l1_pager.tools import FETCH_EVICTED_PAGE_NAME
from l1_pager.wrap import WrapModelCallMiddleware, wrap_model_call

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NEEDLE_TXN_ID = "TXN-PHASE3-9999"
NEEDLE_AMOUNT = 88_123.45


def _needle_content() -> str:
    blob = json.dumps({"transaction_id": NEEDLE_TXN_ID, "amount": NEEDLE_AMOUNT, "status": "settled"})
    padding = "\n".join(f"AUDIT [{i:04d}] record" for i in range(150))
    return blob + "\n\n" + padding


def _build_msgs(num_turns: int) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    msgs.append(SystemMessage(content="Financial assistant.", id=str(uuid4())))
    tc_id = str(uuid4())
    msgs.append(HumanMessage(content="Look up transaction.", id=str(uuid4())))
    msgs.append(AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {}, "id": tc_id}],
        id=str(uuid4()),
    ))
    msgs.append(ToolMessage(
        content=_needle_content(),
        tool_call_id=tc_id,
        name="lookup",
        id=str(uuid4()),
    ))
    for i in range(1, num_turns):
        msgs.append(HumanMessage(content=f"Q{i}", id=str(uuid4())))
        msgs.append(AIMessage(content=f"A{i}.", id=str(uuid4())))
    return msgs


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._r = list(responses)
        self._i = 0
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages, **kw) -> AIMessage:
        self.calls.append(list(messages))
        if self._i >= len(self._r):
            raise RuntimeError(f"No more scripted responses (call #{self._i + 1})")
        r = self._r[self._i]
        self._i += 1
        return r

    def invoke(self, messages, **kw) -> AIMessage:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(messages))

    def bind_tools(self, tools, **kw):
        return self

    @property
    def call_count(self) -> int:
        return self._i


# ---------------------------------------------------------------------------
# should_continue router
# ---------------------------------------------------------------------------


class TestShouldContinue:
    def _state(self, last: BaseMessage) -> dict:
        return {"messages": [last]}

    def test_no_tool_calls_returns_end(self):
        ai = AIMessage(content="Done.", id="1")
        assert should_continue(self._state(ai)) == END

    def test_real_tool_calls_returns_tools(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "tc1"}],
            id="1",
        )
        assert should_continue(self._state(ai)) == "tools"

    def test_page_fault_only_returns_end(self):
        """fetch_evicted_page is consumed by L1Pager inside the agent node."""
        ai = AIMessage(
            content="",
            tool_calls=[{"name": FETCH_EVICTED_PAGE_NAME, "args": {"page_id": "abc"}, "id": "tc1"}],
            id="1",
        )
        assert should_continue(self._state(ai)) == END

    def test_mixed_real_and_pager_tool_returns_tools(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": FETCH_EVICTED_PAGE_NAME, "args": {"page_id": "abc"}, "id": "tc1"},
                {"name": "search", "args": {}, "id": "tc2"},
            ],
            id="1",
        )
        assert should_continue(self._state(ai)) == "tools"

    def test_empty_messages_returns_end(self):
        assert should_continue({"messages": []}) == END


# ---------------------------------------------------------------------------
# create_l1_pager_node
# ---------------------------------------------------------------------------


class TestCreateL1PagerNode:
    @pytest.mark.asyncio
    async def test_node_increments_turn_count(self):
        config = EvictionConfig()
        heap = build_heap(config)
        model = _ScriptedModel([AIMessage(content="hi", id=str(uuid4()))])
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        state = {"messages": [HumanMessage(content="hello", id=str(uuid4()))], "l1_turn_count": 3}
        result = await node(state)

        assert result["l1_turn_count"] == 4

    @pytest.mark.asyncio
    async def test_node_returns_messages_with_ai_response(self):
        config = EvictionConfig()
        heap = build_heap(config)
        ai_resp = AIMessage(content="Hello!", id=str(uuid4()))
        model = _ScriptedModel([ai_resp])
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        state = {"messages": [HumanMessage(content="hi", id=str(uuid4()))]}
        result = await node(state)

        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        assert last.content == "Hello!"

    @pytest.mark.asyncio
    async def test_node_records_evictions_in_state(self):
        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        msgs = _build_msgs(num_turns=10)  # turn-0 tool output will be swept
        ai_resp = AIMessage(content="done", id=str(uuid4()))
        model = _ScriptedModel([ai_resp])
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        state = {"messages": msgs, "l1_turn_count": 10}
        result = await node(state)

        assert result.get("l1_eviction_log"), "Expected eviction records when sweep occurred"
        assert result["l1_tokens_freed_total"] > 0

    @pytest.mark.asyncio
    async def test_node_handles_missing_l1_state_fields(self):
        """Node must not crash when graph state has no L1 fields yet."""
        config = EvictionConfig()
        heap = build_heap(config)
        model = _ScriptedModel([AIMessage(content="ok", id=str(uuid4()))])
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        # Minimal state — no l1_* keys
        state = {"messages": [HumanMessage(content="hi", id=str(uuid4()))]}
        result = await node(state)

        assert result["l1_turn_count"] == 1
        assert result["l1_page_fault_total"] == 0


# ---------------------------------------------------------------------------
# Full StateGraph integration
# ---------------------------------------------------------------------------


class TestStateGraphIntegration:
    """Build a real StateGraph with create_l1_pager_node and run it."""

    def _build_graph(self, model: _ScriptedModel, config: EvictionConfig) -> Any:
        heap = build_heap(config)
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        builder = StateGraph(L1PagerGraphState)
        builder.add_node("agent", node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        return builder.compile()

    @pytest.mark.asyncio
    async def test_graph_single_turn(self):
        config = EvictionConfig()
        ai = AIMessage(content="Answer!", id=str(uuid4()))
        model = _ScriptedModel([ai])
        graph = self._build_graph(model, config)

        result = await graph.ainvoke({
            "messages": [HumanMessage(content="Hello?", id=str(uuid4()))],
        })

        assert result["messages"][-1].content == "Answer!"
        assert result["l1_turn_count"] == 1

    @pytest.mark.asyncio
    async def test_graph_state_accumulates_over_multiple_invoke_calls(self):
        """Simulate sequential node invocations — turn count must accumulate."""
        config = EvictionConfig()
        model = _ScriptedModel([
            AIMessage(content="A1", id=str(uuid4())),
            AIMessage(content="A2", id=str(uuid4())),
        ])
        graph = self._build_graph(model, config)

        state1 = await graph.ainvoke({
            "messages": [HumanMessage(content="Q1", id=str(uuid4()))],
        })
        state2 = await graph.ainvoke({
            "messages": state1["messages"] + [HumanMessage(content="Q2", id=str(uuid4()))],
            "l1_turn_count": state1["l1_turn_count"],
        })

        assert state2["l1_turn_count"] == 2


# ---------------------------------------------------------------------------
# create_l1_pager_react_agent — full ReAct loop
# ---------------------------------------------------------------------------


class TestCreateL1PagerReactAgent:
    @pytest.mark.asyncio
    async def test_agent_single_turn_no_tools(self):
        config = EvictionConfig()
        ai = AIMessage(content="Done.", id=str(uuid4()))
        model = _ScriptedModel([ai])

        agent = create_l1_pager_react_agent(model, tools=[], config=config)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Hello", id=str(uuid4()))],
        })

        assert result["messages"][-1].content == "Done."

    @pytest.mark.asyncio
    async def test_agent_calls_real_tool_and_loops(self):
        """
        Model first calls a real tool, then returns a final answer.
        L1Pager must not interfere with the normal tool loop.
        """
        config = EvictionConfig()

        @tool
        def echo(text: str) -> str:
            """Echo the text back."""
            return f"ECHO: {text}"

        tc_id = str(uuid4())
        call_1 = AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "hello"}, "id": tc_id}],
            id=str(uuid4()),
        )
        call_2 = AIMessage(content="The echo said: ECHO: hello", id=str(uuid4()))

        model = _ScriptedModel([call_1, call_2])
        agent = create_l1_pager_react_agent(model, tools=[echo], config=config)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Echo hello", id=str(uuid4()))],
        })

        assert "ECHO: hello" in result["messages"][-1].content

    @pytest.mark.asyncio
    async def test_agent_resolves_page_fault_in_react_loop(self):
        """
        Full battle test through the graph:
          50 turns → sweep → pointer in state → model fires fetch_evicted_page
          → L1Pager re-injects → model recalls the needle value.
        """
        from l1_pager.sweep import SweepPhase
        from l1_pager.schema import parse_pointer

        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)

        msgs = _build_msgs(num_turns=50)
        # Find the ToolMessage id that will be swept
        tool_msg_id = next(m.id for m in msgs if isinstance(m, ToolMessage) and m.name == "lookup")

        # Pre-sweep so we know the page_id
        sweep = SweepPhase(config, heap)
        pre = await sweep.run(msgs, current_turn=50)
        pointer_msg = next(m for m in pre.updated_messages if m.id == tool_msg_id)
        parsed = parse_pointer(pointer_msg.content)
        assert parsed is not None
        page_id, _ = parsed

        # Build scripted model: recognise pointer → page fault → answer
        fault_resp = AIMessage(
            content="",
            tool_calls=[{"name": FETCH_EVICTED_PAGE_NAME, "args": {"page_id": page_id}, "id": str(uuid4())}],
            id=str(uuid4()),
        )
        answer_resp = AIMessage(
            content=f"The amount was ${NEEDLE_AMOUNT:,.2f} and TXN ID is {NEEDLE_TXN_ID}.",
            id=str(uuid4()),
        )
        model = _ScriptedModel([fault_resp, answer_resp])

        # Wire L1Pager with the pre-populated heap
        pager = L1Pager(model=model, heap=heap, config=config)
        node = create_l1_pager_node(pager)

        builder = StateGraph(L1PagerGraphState)
        builder.add_node("agent", node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile()

        # Add user question to swept messages
        question = HumanMessage(content="What was that amount?", id=str(uuid4()))
        input_msgs = pre.updated_messages + [question]

        result = await graph.ainvoke({
            "messages": input_msgs,
            "l1_turn_count": 50,
        })

        last = result["messages"][-1]
        assert "88,123.45" in last.content or "88123.45" in last.content
        assert NEEDLE_TXN_ID in last.content
        assert result["l1_page_fault_total"] == 1

    @pytest.mark.asyncio
    async def test_bind_tools_includes_fetch_evicted_page(self):
        """_ModelShim.bind_tools must inject fetch_evicted_page alongside user tools."""
        config = EvictionConfig()
        heap = build_heap(config)

        bound_tools_received: list = []

        class _SpyModel:
            def bind_tools(self, tools, **kw):
                bound_tools_received.extend(tools)
                return self
            async def ainvoke(self, messages, **kw):
                return AIMessage(content="ok", id=str(uuid4()))

        @tool
        def my_tool(x: str) -> str:
            """A user tool."""
            return x

        pager = L1Pager(model=_SpyModel(), heap=heap, config=config)
        shim = pager.as_model()
        shim.bind_tools([my_tool])

        tool_names = {getattr(t, "name", None) for t in bound_tools_received}
        assert "my_tool" in tool_names
        assert FETCH_EVICTED_PAGE_NAME in tool_names


# ---------------------------------------------------------------------------
# WrapModelCallMiddleware
# ---------------------------------------------------------------------------


class TestWrapModelCallMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_ainvoke_returns_ai_message(self):
        config = EvictionConfig()
        heap = build_heap(config)
        model = _ScriptedModel([AIMessage(content="wrapped!", id=str(uuid4()))])
        pager = L1Pager(model=model, heap=heap, config=config)
        middleware = WrapModelCallMiddleware(pager)

        result = await middleware.ainvoke([HumanMessage(content="hi", id=str(uuid4()))])
        assert isinstance(result, AIMessage)
        assert result.content == "wrapped!"

    @pytest.mark.asyncio
    async def test_middleware_bind_tools_forwards_tools(self):
        config = EvictionConfig()
        heap = build_heap(config)
        bound_tools: list = []

        class _SpyModel:
            def bind_tools(self, tools, **kw):
                bound_tools.extend(tools)
                return self
            async def ainvoke(self, messages, **kw):
                return AIMessage(content="ok", id=str(uuid4()))

        @tool
        def calc(x: int) -> int:
            """Calculator."""
            return x * 2

        pager = L1Pager(model=_SpyModel(), heap=heap, config=config)
        middleware = WrapModelCallMiddleware(pager)
        middleware.bind_tools([calc])

        names = {getattr(t, "name", None) for t in bound_tools}
        assert "calc" in names
        assert FETCH_EVICTED_PAGE_NAME in names


# ---------------------------------------------------------------------------
# wrap_model_call functional wrapper
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    @pytest.mark.asyncio
    async def test_wrapped_callable_returns_ai_message(self):
        config = EvictionConfig()
        heap = build_heap(config)
        model = _ScriptedModel([AIMessage(content="result", id=str(uuid4()))])
        pager = L1Pager(model=model, heap=heap, config=config)
        call = wrap_model_call(model, pager)

        result = await call([HumanMessage(content="test", id=str(uuid4()))])
        assert isinstance(result, AIMessage)
        assert result.content == "result"

    @pytest.mark.asyncio
    async def test_wrapped_callable_accepts_current_turn(self):
        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        msgs = _build_msgs(num_turns=10)
        model = _ScriptedModel([AIMessage(content="ok", id=str(uuid4()))])
        pager = L1Pager(model=model, heap=heap, config=config)
        call = wrap_model_call(model, pager)

        # Should not raise even with a large current_turn value
        result = await call(msgs, current_turn=10)
        assert isinstance(result, AIMessage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
