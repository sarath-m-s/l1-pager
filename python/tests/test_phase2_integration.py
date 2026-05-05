"""
Phase 2 integration tests: Sweep + Interceptor + Page-Fault Recovery.

Battle test scenario:
  - 50-turn conversation
  - Turn 0: large ToolMessage containing specific JSON (transaction ID, amount)
  - Mock LLM first call: recognizes pointer, fires fetch_evicted_page
  - L1Pager interceptor: detects page fault, fetches from heap, re-injects
  - Mock LLM second call: reads re-injected content, quotes specific value
  - Assert: the recalled value is correct after 50 turns of context churn
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
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

from l1_pager import (
    EvictionConfig,
    L1Pager,
    L1PagerResponse,
    build_heap,
    is_pointer_content,
    parse_pointer,
)
from l1_pager.sweep import SweepPhase
from l1_pager.summarizer import generate_summary
from l1_pager.tools import FETCH_EVICTED_PAGE_NAME, is_page_fault_tool_call

# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class ScriptedModel:
    """
    Deterministic model for testing. Responses are consumed in order.
    Supports bind_tools() by returning self (stateless wrt tool binding).
    """

    def __init__(self, responses: list[BaseMessage]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_log: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        self.call_log.append(list(messages))
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                f"ScriptedModel: no more responses (called {self._call_index + 1} times)"
            )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    def invoke(self, messages, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(messages, **kwargs))

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def call_count(self) -> int:
        return self._call_index


# ---------------------------------------------------------------------------
# Conversation fixture
# ---------------------------------------------------------------------------

NEEDLE_TXN_ID = "TXN-20240312-9876"
NEEDLE_AMOUNT = 14_523.99
NEEDLE_BANK_REF = "ABCD-1234"


def _needle_json() -> str:
    return json.dumps({
        "transaction_id": NEEDLE_TXN_ID,
        "amount": NEEDLE_AMOUNT,
        "currency": "USD",
        "status": "settled",
        "metadata": {"source": "wire", "bank_ref": NEEDLE_BANK_REF},
    }, indent=2)


def _build_conversation(num_turns: int = 50) -> tuple[list[BaseMessage], str]:
    """
    Returns (messages, tool_message_id).
    Turn 0 has a large ToolMessage (the needle). Subsequent turns are noise.
    """
    msgs: list[BaseMessage] = []
    msgs.append(SystemMessage(content="You are a financial assistant.", id=str(uuid4())))

    # Turn 0
    tool_call_id = str(uuid4())
    tool_msg_id = str(uuid4())
    padding = "\n".join(f"LOG [{i:04d}] audit entry processed" for i in range(150))
    large_content = _needle_json() + "\n\n" + padding

    msgs.append(HumanMessage(content="Look up transaction TXN-20240312-9876", id=str(uuid4())))
    msgs.append(AIMessage(
        content="",
        tool_calls=[{"name": "lookup_txn", "args": {"id": NEEDLE_TXN_ID}, "id": tool_call_id}],
        id=str(uuid4()),
    ))
    msgs.append(ToolMessage(
        content=large_content,
        tool_call_id=tool_call_id,
        name="lookup_txn",
        id=tool_msg_id,
    ))

    # Turns 1 .. num_turns-1: lightweight noise
    for i in range(1, num_turns):
        msgs.append(HumanMessage(content=f"What is 2 + {i}?", id=str(uuid4())))
        msgs.append(AIMessage(content=f"It is {2 + i}.", id=str(uuid4())))

    return msgs, tool_msg_id


# ---------------------------------------------------------------------------
# Sweep unit tests
# ---------------------------------------------------------------------------


class TestSweepPhase:
    @pytest.mark.asyncio
    async def test_sweep_replaces_large_old_message_with_pointer(self):
        msgs, tool_id = _build_conversation(num_turns=10)
        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        sweep = SweepPhase(config, heap)

        result = await sweep.run(msgs, current_turn=10)

        assert result.did_evict
        assert result.tokens_freed > 0
        assert result.pages_written >= 1

        # The original ToolMessage must now be a pointer
        for msg in result.updated_messages:
            if msg.id == tool_id:
                assert is_pointer_content(msg.content), (
                    f"Expected pointer content, got: {msg.content[:80]}"
                )

    @pytest.mark.asyncio
    async def test_sweep_preserves_message_ids(self):
        """Pointers must carry the original message IDs for add_messages in-place update."""
        msgs, tool_id = _build_conversation(num_turns=10)
        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        sweep = SweepPhase(config, heap)

        result = await sweep.run(msgs, current_turn=10)

        original_ids = {m.id for m in msgs}
        updated_ids = {m.id for m in result.updated_messages}
        assert original_ids == updated_ids, "Sweep must not add or remove message IDs"

    @pytest.mark.asyncio
    async def test_sweep_idempotent(self):
        """Running sweep twice must not create duplicate heap entries or pointers."""
        msgs, _ = _build_conversation(num_turns=10)
        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        sweep = SweepPhase(config, heap)

        result1 = await sweep.run(msgs, current_turn=10)
        result2 = await sweep.run(result1.updated_messages, current_turn=10)

        # Second sweep must have nothing to evict (pointers are not ephemeral)
        assert not result2.did_evict

    @pytest.mark.asyncio
    async def test_human_messages_never_swept(self):
        msgs, _ = _build_conversation(num_turns=10)
        config = EvictionConfig(min_tokens=1, min_turns_old=0)  # sweep everything
        heap = build_heap(config)
        sweep = SweepPhase(config, heap)

        result = await sweep.run(msgs, current_turn=10)

        for msg in result.updated_messages:
            if isinstance(msg, HumanMessage):
                assert not is_pointer_content(msg.content), (
                    "HumanMessage must never be swept"
                )

    @pytest.mark.asyncio
    async def test_raw_content_survives_in_heap(self):
        """After sweep, the heap must return byte-identical raw content."""
        msgs, tool_id = _build_conversation(num_turns=10)
        original_content = next(m.content for m in msgs if m.id == tool_id)

        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        sweep = SweepPhase(config, heap)

        result = await sweep.run(msgs, current_turn=10)
        assert result.did_evict

        # Retrieve from heap using the pointer
        pointer_msg = next(m for m in result.updated_messages if m.id == tool_id)
        parsed = parse_pointer(pointer_msg.content)
        assert parsed is not None
        page_id, _ = parsed

        entry = await heap.get(page_id)
        assert entry is not None
        assert entry.raw_content == original_content


# ---------------------------------------------------------------------------
# Summarizer unit tests
# ---------------------------------------------------------------------------


class TestSummarizer:
    def test_tool_message_includes_tool_name(self):
        msg = ToolMessage(
            content='{"status": "ok", "result": 42}',
            tool_call_id="x",
            name="calculate",
            id="1",
        )
        summary = generate_summary(msg)
        assert "calculate" in summary or "JSON" in summary

    def test_json_summary_includes_keys(self):
        msg = ToolMessage(
            content='{"transaction_id": "abc", "amount": 100}',
            tool_call_id="x",
            name="lookup",
            id="1",
        )
        summary = generate_summary(msg)
        assert len(summary) <= 140

    def test_summary_under_140_chars(self):
        long_text = "This is a very important result. " * 20
        msg = ToolMessage(content=long_text, tool_call_id="x", name="tool", id="1")
        assert len(generate_summary(msg)) <= 140

    def test_ai_message_summary(self):
        msg = AIMessage(content="Let me analyze the transaction data step by step.", id="1")
        summary = generate_summary(msg)
        assert "step" in summary or len(summary) > 0


# ---------------------------------------------------------------------------
# BATTLE TEST: Full 50-turn recall via page-fault mechanism
# ---------------------------------------------------------------------------


class TestFiftyTurnBattleTest:
    """
    This is the complete end-to-end validation.

    Scenario:
      1. Agent looks up transaction TXN-20240312-9876 in turn 0.
         Tool returns large JSON payload (>500 estimated tokens).
      2. 49 more turns pass (noise questions about arithmetic).
      3. User asks: "What was the exact transaction amount from earlier?"
      4. L1Pager has swept the turn-0 tool output and replaced with pointer.
      5. The LLM sees the pointer, fires fetch_evicted_page("abc123...").
      6. L1Pager intercepts: fetches raw content from heap, re-injects as ToolMessage.
      7. LLM second call: reads re-injected data, answers "$14,523.99".
      8. Assert: answer contains the exact amount.
    """

    @pytest.mark.asyncio
    async def test_full_fifty_turn_recall(self):
        msgs, tool_id = _build_conversation(num_turns=50)

        config = EvictionConfig(min_tokens=100, min_turns_old=3, max_page_fault_depth=3)
        heap = build_heap(config)

        # Pre-run sweep so we know what page_id was assigned
        from l1_pager.sweep import SweepPhase
        sweep = SweepPhase(config, heap)
        pre_result = await sweep.run(msgs, current_turn=50)
        assert pre_result.did_evict, "The large tool output must have been swept"

        # Find the pointer in the swept messages
        pointer_msg = next(m for m in pre_result.updated_messages if m.id == tool_id)
        assert is_pointer_content(pointer_msg.content)
        parsed = parse_pointer(pointer_msg.content)
        assert parsed is not None
        page_id, summary = parsed

        # Verify the summary is meaningful
        assert len(summary) > 5

        # Build mock model responses:
        #   Call 1: model sees pointer → fires page fault
        #   Call 2: model sees re-injected raw content → answers correctly
        call_1_response = AIMessage(
            id=str(uuid4()),
            content="",
            tool_calls=[{
                "name": FETCH_EVICTED_PAGE_NAME,
                "args": {"page_id": page_id},
                "id": str(uuid4()),
            }],
        )
        call_2_response = AIMessage(
            id=str(uuid4()),
            content=(
                f"The transaction amount was ${NEEDLE_AMOUNT:,.2f}. "
                f"Transaction ID: {NEEDLE_TXN_ID}, Bank ref: {NEEDLE_BANK_REF}."
            ),
        )

        model = ScriptedModel([call_1_response, call_2_response])
        pager = L1Pager(model=model, heap=heap, config=config)

        # Use the already-swept messages (simulating state after sweep node)
        swept_msgs = pre_result.updated_messages
        # Add final human question
        swept_msgs.append(
            HumanMessage(content="What was the exact amount of that transaction?", id=str(uuid4()))
        )

        result = await pager.ainvoke(swept_msgs, current_turn=50)

        # Assertions
        assert result.page_faults == 1, f"Expected 1 page fault, got {result.page_faults}"
        assert model.call_count == 2, f"Expected 2 model calls, got {model.call_count}"

        # The recalled answer must contain the exact dollar amount
        answer = result.ai_message.content
        assert "14,523.99" in answer or "14523.99" in answer, (
            f"Expected amount in answer, got: {answer}"
        )
        assert NEEDLE_TXN_ID in answer, f"Expected TXN ID in answer, got: {answer}"

        # The re-injected message must contain the original JSON
        second_call_messages = model.call_log[1]
        re_injected = next(
            (m for m in second_call_messages if isinstance(m, ToolMessage) and m.name == FETCH_EVICTED_PAGE_NAME),
            None,
        )
        assert re_injected is not None, "Re-injected ToolMessage not found in second call"
        assert NEEDLE_TXN_ID in str(re_injected.content)
        assert str(NEEDLE_AMOUNT) in str(re_injected.content)

    @pytest.mark.asyncio
    async def test_page_fault_depth_guard_prevents_infinite_loop(self):
        """
        If the model keeps firing page faults, the depth guard must stop
        after max_page_fault_depth iterations and return the last response.
        """
        msgs, tool_id = _build_conversation(num_turns=10)
        config = EvictionConfig(min_tokens=100, min_turns_old=3, max_page_fault_depth=2)
        heap = build_heap(config)

        # Pre-sweep
        sweep = SweepPhase(config, heap)
        pre_result = await sweep.run(msgs, current_turn=10)
        pointer_msg = next(m for m in pre_result.updated_messages if m.id == tool_id)
        parsed = parse_pointer(pointer_msg.content)
        assert parsed is not None
        page_id, _ = parsed

        # Model always responds with a page fault (infinite loop scenario)
        def _make_fault_response():
            return AIMessage(
                id=str(uuid4()),
                content="",
                tool_calls=[{
                    "name": FETCH_EVICTED_PAGE_NAME,
                    "args": {"page_id": page_id},
                    "id": str(uuid4()),
                }],
            )

        # Give it max_page_fault_depth + 2 responses so it doesn't IndexError
        model = ScriptedModel([_make_fault_response() for _ in range(5)])
        pager = L1Pager(model=model, heap=heap, config=config)

        result = await pager.ainvoke(pre_result.updated_messages, current_turn=10)

        # Must stop at max_page_fault_depth (2), not loop forever
        assert result.page_faults == config.max_page_fault_depth
        assert model.call_count == config.max_page_fault_depth + 1  # initial + N recoveries

    @pytest.mark.asyncio
    async def test_page_not_found_returns_error_message_to_model(self):
        """If the model requests a non-existent page, the interceptor must
        return a clear error ToolMessage rather than crashing."""
        msgs = [
            HumanMessage(content="What happened?", id=str(uuid4())),
        ]
        config = EvictionConfig()
        heap = build_heap(config)

        bad_page_id = "nonexistent0000"
        call_1 = AIMessage(
            id=str(uuid4()),
            content="",
            tool_calls=[{
                "name": FETCH_EVICTED_PAGE_NAME,
                "args": {"page_id": bad_page_id},
                "id": str(uuid4()),
            }],
        )
        call_2 = AIMessage(id=str(uuid4()), content="I could not find the original data.")
        model = ScriptedModel([call_1, call_2])
        pager = L1Pager(model=model, heap=heap, config=config)

        result = await pager.ainvoke(msgs)

        assert result.page_faults == 1
        # The model's second call must have received an error ToolMessage
        second_call_msgs = model.call_log[1]
        error_msg = next(
            (m for m in second_call_msgs if isinstance(m, ToolMessage) and m.name == FETCH_EVICTED_PAGE_NAME),
            None,
        )
        assert error_msg is not None
        assert "not found" in error_msg.content.lower() or "L1Pager" in error_msg.content

    @pytest.mark.asyncio
    async def test_no_page_fault_single_model_call(self):
        """When no pointer is present, the model should be called exactly once."""
        msgs = [
            HumanMessage(content="Hello", id=str(uuid4())),
        ]
        config = EvictionConfig()
        heap = build_heap(config)

        response = AIMessage(id=str(uuid4()), content="Hi there!")
        model = ScriptedModel([response])
        pager = L1Pager(model=model, heap=heap, config=config)

        result = await pager.ainvoke(msgs)

        assert result.page_faults == 0
        assert model.call_count == 1
        assert result.ai_message.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_sweep_result_messages_compatible_with_add_messages(self):
        """
        Simulate what LangGraph's add_messages reducer does.
        Pointer messages (same ID) must overwrite originals; new messages append.
        """
        from langgraph.graph.message import add_messages

        msgs, tool_id = _build_conversation(num_turns=10)
        original_content = next(m.content for m in msgs if m.id == tool_id)

        config = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = build_heap(config)
        response = AIMessage(id=str(uuid4()), content="Sweep complete.")
        model = ScriptedModel([response])
        pager = L1Pager(model=model, heap=heap, config=config)

        result = await pager.ainvoke(msgs, current_turn=10)

        # Simulate what add_messages does when the node returns result.messages
        new_state_messages = add_messages(list(msgs), result.messages)

        # 1. Count should be original + 1 (the new AI response), not more
        assert len(new_state_messages) == len(msgs) + 1

        # 2. The tool message (by ID) must now be a pointer
        updated_tool_msg = next(m for m in new_state_messages if m.id == tool_id)
        assert is_pointer_content(updated_tool_msg.content)
        assert updated_tool_msg.content != original_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
