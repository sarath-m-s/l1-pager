"""
L1-Pager: LangGraph integration — node factory and graph builder.

Three public APIs:

1. ``create_l1_pager_node(pager)``
   Returns a plain async node function compatible with any StateGraph.
   Use this when you want full control over your graph structure.

2. ``create_l1_pager_react_agent(model, tools, ...)``
   Drop-in replacement for ``langgraph.prebuilt.create_react_agent`` that
   wires L1Pager into the agent + tool loop automatically.
   The ``fetch_evicted_page`` page-fault is handled *inside* the agent node,
   so the ToolNode never sees it.

3. ``should_continue(state, fetch_tool_name)``
   Router that filters out L1Pager's internal tool calls so they never reach
   the ToolNode.

Routing invariant
-----------------
  agent_node  →  should_continue  →  "tools"  →  tool_node  →  agent_node
                                  →  END

Only *real* tool calls (not fetch_evicted_page) trigger the "tools" branch.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from .heap import AbstractHeap, build_heap
from .interceptor import L1Pager, L1PagerResponse
from .schema import EvictionConfig
from .tools import FETCH_EVICTED_PAGE_NAME

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal graph state (standalone — no L1PagerState mixin required)
# ---------------------------------------------------------------------------


class L1PagerGraphState(TypedDict, total=False):
    """
    A self-contained graph state that includes L1Pager observability fields.

    Users can extend this with their own fields:

        class MyState(L1PagerGraphState):
            user_id: str
    """

    messages: Annotated[list[BaseMessage], add_messages]
    l1_turn_count: int
    l1_eviction_log: list[dict]
    l1_page_fault_total: int
    l1_tokens_freed_total: int


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def create_l1_pager_node(pager: L1Pager) -> Callable:
    """
    Return an async LangGraph node function that runs the full L1Pager pipeline.

    The node:
      1. Sweeps old large messages → writes to heap → injects pointers
      2. Calls the model
      3. Handles any fetch_evicted_page page faults inline
      4. Returns updated messages + observability counters

    Usage::

        pager = L1Pager(model=model, heap=heap)
        builder.add_node("agent", create_l1_pager_node(pager))
    """

    async def _node(state: dict) -> dict:
        messages: list[BaseMessage] = state.get("messages", [])
        turn: int = state.get("l1_turn_count", 0)

        result: L1PagerResponse = await pager.ainvoke(messages, current_turn=turn)

        updates: dict[str, Any] = {
            "messages": result.messages,
            "l1_turn_count": turn + 1,
            "l1_page_fault_total": state.get("l1_page_fault_total", 0) + result.page_faults,
            "l1_tokens_freed_total": state.get("l1_tokens_freed_total", 0) + result.tokens_freed,
        }

        if result.eviction_records:
            updates["l1_eviction_log"] = [r.to_dict() for r in result.eviction_records]

        if result.page_faults:
            log.info(
                "L1Pager: %d page fault(s) resolved, %d tokens freed this turn",
                result.page_faults,
                result.tokens_freed,
            )

        return updates

    return _node


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def should_continue(
    state: dict,
) -> Literal["tools", "__end__"]:
    """
    Conditional edge: route to "tools" if the last message has real tool calls,
    otherwise end.

    "Real" = any tool call that is NOT fetch_evicted_page (that one is already
    consumed by L1Pager inside the agent node and never reaches ToolNode).
    """
    messages = state.get("messages", [])
    if not messages:
        return END  # type: ignore[return-value]

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return END  # type: ignore[return-value]

    tool_calls = getattr(last, "tool_calls", None) or []
    real_calls = [tc for tc in tool_calls if tc.get("name") != FETCH_EVICTED_PAGE_NAME]

    return "tools" if real_calls else END  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_l1_pager_react_agent(
    model: Any,
    tools: list[BaseTool],
    *,
    heap: AbstractHeap | None = None,
    config: EvictionConfig | None = None,
    state_schema: type | None = None,
    system_prompt: str | None = None,
) -> Any:
    """
    Drop-in replacement for ``langgraph.prebuilt.create_react_agent`` with
    L1Pager context GC built in.

    Key difference from the prebuilt version:
    - The agent node runs the full L1Pager sweep + page-fault recovery pipeline.
    - ``fetch_evicted_page`` is intercepted *before* ToolNode sees it.
    - ``should_continue`` filters out L1Pager tool calls from the routing decision.

    Args:
        model: Any LangChain chat model (ChatOpenAI, ChatAnthropic, etc.)
        tools: User-defined tools. Do NOT include fetch_evicted_page — it is
               added automatically.
        heap: Heap backend. Defaults to InMemoryHeap.
        config: EvictionConfig. Defaults to sensible defaults.
        state_schema: Optional custom TypedDict. Defaults to L1PagerGraphState.
        system_prompt: Optional system message content prepended to every call.

    Returns:
        A compiled LangGraph ``CompiledGraph`` with the same interface as
        ``create_react_agent`` (call ``.invoke()``, ``.ainvoke()``,
        ``.stream()``).
    """
    _config = config or EvictionConfig()
    _heap = heap or build_heap(_config, backend="memory")
    _state = state_schema or L1PagerGraphState

    # Bind user tools + fetch_evicted_page to the model upfront so the LLM
    # knows it can call fetch_evicted_page to resolve page-fault pointers
    from .tools import fetch_evicted_page_tool
    user_tool_names = {getattr(t, "name", None) for t in tools}
    all_tools = list(tools)
    if FETCH_EVICTED_PAGE_NAME not in user_tool_names:
        all_tools.append(fetch_evicted_page_tool)

    pager = L1Pager(model=model, heap=_heap, config=_config)
    # Override the bound model with all tools
    pager._bound_model = model.bind_tools(all_tools)

    agent_node = create_l1_pager_node(pager)

    # ToolNode only receives the *user* tools — never fetch_evicted_page
    user_tools_only = [t for t in tools if getattr(t, "name", None) != FETCH_EVICTED_PAGE_NAME]
    tool_node = ToolNode(user_tools_only)

    # Optionally prepend system message
    if system_prompt:
        from langchain_core.messages import SystemMessage

        _system = SystemMessage(content=system_prompt)

        async def _agent_with_system(state: dict) -> dict:
            messages = state.get("messages", [])
            if not messages or not isinstance(messages[0], SystemMessage):
                state = dict(state)
                state["messages"] = [_system] + list(messages)
            return await agent_node(state)

        final_agent_node = _agent_with_system
    else:
        final_agent_node = agent_node

    builder = StateGraph(_state)
    builder.add_node("agent", final_agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()
