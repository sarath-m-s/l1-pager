"""
Phase 1 validation: Mark phase + InMemoryHeap.

Critical test case: a model must be able to recall a specific JSON value
from a tool output that was swept 50 turns ago. We verify:
  1. The message is correctly marked (criteria satisfied)
  2. A pointer is generated in the correct format
  3. The raw content survives in the heap and is retrievable
  4. Messages that do NOT meet criteria are never marked
"""
import asyncio
import json
import time
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from l1_pager.schema import (
    EvictionConfig,
    EvictionPolicy,
    PageHeapEntry,
    compute_page_id,
    is_pointer_content,
    make_page_pointer_message,
    parse_pointer,
    render_pointer,
)
from l1_pager.mark import (
    MarkPhase,
    compute_turn_ages,
    estimate_tokens,
    is_ephemeral,
    importance_score,
)
from l1_pager.heap import InMemoryHeap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conversation(num_turns: int, include_large_tool: bool = True) -> list:
    """
    Build a synthetic conversation of `num_turns` turns.

    Turn 0 injects a large ToolMessage (the "needle" we will recall later).
    Subsequent turns are small H+AI exchanges.
    """
    msgs = []
    msgs.append(SystemMessage(content="You are a helpful assistant.", id=str(uuid4())))

    # Turn 0: large tool output with a specific JSON payload (the needle)
    needle_json = json.dumps({
        "transaction_id": "TXN-20240312-9876",
        "amount": 14_523.99,
        "currency": "USD",
        "status": "settled",
        "metadata": {"source": "wire", "bank_ref": "ABCD-1234"},
    })
    # Pad to >500 tokens: ~2000 chars of repetitive log lines
    padding = "\n".join([f"LOG [{i:04d}] processing..." for i in range(120)])
    large_content = needle_json + "\n\n" + padding

    tool_call_id = str(uuid4())
    msgs.append(HumanMessage(content="Check the transaction status.", id=str(uuid4())))
    msgs.append(AIMessage(
        content="",
        tool_calls=[{"name": "check_transaction", "args": {}, "id": tool_call_id}],
        id=str(uuid4()),
    ))
    if include_large_tool:
        msgs.append(ToolMessage(
            content=large_content,
            tool_call_id=tool_call_id,
            name="check_transaction",
            id=str(uuid4()),
        ))

    # Turns 1..num_turns-1: lightweight exchanges
    for i in range(1, num_turns):
        msgs.append(HumanMessage(content=f"Turn {i}: what is the weather?", id=str(uuid4())))
        msgs.append(AIMessage(content=f"Turn {i}: it is sunny.", id=str(uuid4())))

    return msgs


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_string(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_empty(self):
        assert estimate_tokens("") >= 1

    def test_list_of_blocks(self):
        blocks = [{"text": "a" * 400}, {"text": "b" * 400}]
        assert estimate_tokens(blocks) == 200

    def test_dict_fallback(self):
        assert estimate_tokens({"key": "value"}) >= 1


# ---------------------------------------------------------------------------
# compute_turn_ages
# ---------------------------------------------------------------------------

class TestComputeTurnAges:
    def test_basic_structure(self):
        msgs = [
            HumanMessage(content="h1", id="1"),
            AIMessage(content="a1", id="2"),
            HumanMessage(content="h2", id="3"),
            AIMessage(content="a2", id="4"),
            HumanMessage(content="h3", id="5"),
            AIMessage(content="a3", id="6"),
        ]
        ages = compute_turn_ages(msgs)
        # current turn is 2; turn 0 msgs have age 2, turn 1 have age 1, turn 2 have age 0
        assert ages == [2, 2, 1, 1, 0, 0]

    def test_tool_in_middle(self):
        msgs = [
            HumanMessage(content="h1", id="1"),
            AIMessage(content="", id="2"),
            ToolMessage(content="result", tool_call_id="x", id="3"),
            HumanMessage(content="h2", id="4"),
            AIMessage(content="ok", id="5"),
        ]
        ages = compute_turn_ages(msgs)
        assert ages[3] == 0  # h2 is current turn
        assert ages[0] == 1  # h1 is one turn old


# ---------------------------------------------------------------------------
# is_ephemeral
# ---------------------------------------------------------------------------

class TestIsEphemeral:
    def test_tool_message_is_ephemeral(self):
        msg = ToolMessage(content="data", tool_call_id="x", id="1")
        assert is_ephemeral(msg)

    def test_human_never_ephemeral(self):
        assert not is_ephemeral(HumanMessage(content="hi", id="1"))

    def test_system_never_ephemeral(self):
        assert not is_ephemeral(SystemMessage(content="sys", id="1"))

    def test_pure_dispatcher_ai_not_ephemeral(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "fn", "args": {}, "id": "tc1"}],
            id="1",
        )
        assert not is_ephemeral(msg)

    def test_ai_with_content_is_ephemeral(self):
        msg = AIMessage(content="Let me think through this step by step...", id="1")
        assert is_ephemeral(msg)

    def test_already_swept_tool_message_not_ephemeral(self):
        pointer = render_pointer("abc123", "Transaction data from turn 0")
        msg = ToolMessage(content=pointer, tool_call_id="x", id="1")
        assert not is_ephemeral(msg)


# ---------------------------------------------------------------------------
# MarkPhase
# ---------------------------------------------------------------------------

class TestMarkPhase:
    def test_marks_large_old_tool_output(self):
        msgs = _make_conversation(num_turns=10)
        cfg = EvictionConfig(min_tokens=500, min_turns_old=3)
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)

        # The large ToolMessage from turn 0 should be marked
        assert len(candidates) >= 1
        tool_candidate = next(
            (c for c in candidates if isinstance(c.message, ToolMessage)), None
        )
        assert tool_candidate is not None
        assert tool_candidate.token_count > 500
        assert tool_candidate.turn_age > 3

    def test_does_not_mark_recent_messages(self):
        msgs = _make_conversation(num_turns=4)
        cfg = EvictionConfig(min_tokens=500, min_turns_old=3)
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)

        # Turn 0 tool output is 3 turns old — borderline, must NOT be marked (> not >=)
        for c in candidates:
            assert c.turn_age > 3

    def test_does_not_mark_small_messages(self):
        msgs = [
            HumanMessage(content="hello", id="1"),
            AIMessage(content="world", id="2"),
            ToolMessage(content="tiny result", tool_call_id="x", id="3"),
            HumanMessage(content="ok", id="4"),
            AIMessage(content="sure", id="5"),
        ] * 5  # repeat to age them
        cfg = EvictionConfig(min_tokens=500, min_turns_old=1)
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)
        # All tool messages are tiny (<500 tokens)
        for c in candidates:
            assert c.token_count > 500

    def test_lru_sort_order(self):
        msgs = _make_conversation(num_turns=20)
        cfg = EvictionConfig(policy=EvictionPolicy.LRU, min_tokens=100, min_turns_old=1)
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)
        if len(candidates) >= 2:
            # Oldest first
            assert candidates[0].turn_age >= candidates[1].turn_age

    def test_importance_sort_order(self):
        msgs = _make_conversation(num_turns=20)
        cfg = EvictionConfig(policy=EvictionPolicy.IMPORTANCE, min_tokens=100, min_turns_old=1)
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)
        if len(candidates) >= 2:
            # Least important first
            assert candidates[0].importance_score <= candidates[1].importance_score


# ---------------------------------------------------------------------------
# Pointer round-trip
# ---------------------------------------------------------------------------

class TestPointerRoundTrip:
    def test_render_and_parse(self):
        page_id = "deadbeef12345678"
        summary = "Transaction TXN-20240312-9876 settled for $14,523.99"
        rendered = render_pointer(page_id, summary)

        assert rendered.startswith("<PAGE_FAULT_ID:")
        result = parse_pointer(rendered)
        assert result is not None
        pid, summ = result
        assert pid == page_id
        assert summ == summary

    def test_is_pointer_content(self):
        rendered = render_pointer("abc123ef45678901", "some summary")
        assert is_pointer_content(rendered)
        assert not is_pointer_content("regular message text")
        assert not is_pointer_content(None)

    def test_make_page_pointer_message_preserves_id(self):
        original = ToolMessage(
            content="big data blob " * 200,
            tool_call_id="tc_001",
            name="get_data",
            id="msg-original-001",
        )
        page_id = compute_page_id(original)
        pointer_msg = make_page_pointer_message(original, page_id, "Data blob summary", 5)

        # CRITICAL: same id so add_messages reducer does in-place update
        assert pointer_msg.id == original.id
        assert is_pointer_content(pointer_msg.content)
        assert isinstance(pointer_msg, ToolMessage)
        assert pointer_msg.tool_call_id == original.tool_call_id

    def test_make_page_pointer_message_ai(self):
        original = AIMessage(content="scratchpad " * 200, id="msg-ai-001")
        page_id = compute_page_id(original)
        pointer_msg = make_page_pointer_message(original, page_id, "AI scratchpad summary", 3)
        assert pointer_msg.id == original.id
        assert isinstance(pointer_msg, AIMessage)


# ---------------------------------------------------------------------------
# CRITICAL: 50-turn recall scenario
# ---------------------------------------------------------------------------

class TestFiftyTurnRecall:
    """
    Simulates the core product promise: a model CAN recall a specific JSON
    value from a ToolMessage that was swept 50 turns ago.

    This test covers Phase 1 only (mark + heap store/retrieve). The actual
    page-fault re-injection loop is Phase 3.
    """

    @pytest.mark.asyncio
    async def test_needle_survives_fifty_turns(self):
        # 1. Build a 50-turn conversation with a large tool output at turn 0
        msgs = _make_conversation(num_turns=50)
        cfg = EvictionConfig(min_tokens=100, min_turns_old=3)
        heap = InMemoryHeap(cfg)

        # 2. Mark phase: find the large tool output
        phase = MarkPhase(cfg)
        candidates = phase.scan(msgs)
        assert len(candidates) >= 1

        needle_candidate = candidates[0]
        original_msg = needle_candidate.message
        assert isinstance(original_msg, ToolMessage)

        # 3. Store raw content in heap (simulates Sweep phase)
        page_id = compute_page_id(original_msg)
        entry = PageHeapEntry(
            page_id=page_id,
            original_message_id=original_msg.id,
            original_type=type(original_msg).__name__,
            raw_content=original_msg.content,
            tool_name=original_msg.name,
            tool_call_id=original_msg.tool_call_id,
            token_count=needle_candidate.token_count,
            evicted_at_turn=0,
        )
        await heap.put(entry)

        # 4. Replace in message array with pointer
        pointer_msg = make_page_pointer_message(original_msg, page_id, "Transaction check result", 0)
        msgs[needle_candidate.message_index] = pointer_msg

        # 5. Verify the original is gone from the message array
        live_contents = [m.content for m in msgs if isinstance(m, ToolMessage)]
        assert all(is_pointer_content(c) for c in live_contents)

        # 6. Simulate a "page fault": model requests the evicted page 50 turns later
        retrieved = await heap.get(page_id)
        assert retrieved is not None
        assert retrieved.access_count == 1

        # 7. Verify the needle JSON is intact
        raw = retrieved.raw_content
        assert isinstance(raw, str)
        parsed = json.loads(raw.split("\n\n")[0])  # split off the padding
        assert parsed["transaction_id"] == "TXN-20240312-9876"
        assert parsed["amount"] == 14_523.99
        assert parsed["metadata"]["bank_ref"] == "ABCD-1234"

    @pytest.mark.asyncio
    async def test_heap_lru_evicts_oldest_not_needle(self):
        """Heap must not evict the needle when capacity is tight."""
        cfg = EvictionConfig(min_tokens=100, min_turns_old=1, max_heap_entries=5)
        heap = InMemoryHeap(cfg)

        # Put 4 dummy entries
        for i in range(4):
            e = PageHeapEntry(
                page_id=f"dummy{i:012d}",
                original_message_id=str(uuid4()),
                original_type="ToolMessage",
                raw_content=f"dummy content {i}",
                tool_name="dummy",
                tool_call_id=str(uuid4()),
                token_count=600,
                evicted_at_turn=i,
                evicted_at_ts=time.time() - (100 - i),  # oldest = dummy0
            )
            await heap.put(e)

        # Access the needle to make it MRU
        needle_id = "needleabcdef1234"
        needle_entry = PageHeapEntry(
            page_id=needle_id,
            original_message_id=str(uuid4()),
            original_type="ToolMessage",
            raw_content='{"transaction_id": "TXN-NEEDLE"}',
            tool_name="check",
            tool_call_id=str(uuid4()),
            token_count=600,
            evicted_at_turn=0,
        )
        await heap.put(needle_entry)
        await heap.get(needle_id)  # touch — makes it MRU

        # Now add a 6th entry to trigger eviction of LRU (dummy0)
        overflow = PageHeapEntry(
            page_id="overflow00000001",
            original_message_id=str(uuid4()),
            original_type="ToolMessage",
            raw_content="overflow",
            tool_name="x",
            tool_call_id=str(uuid4()),
            token_count=600,
            evicted_at_turn=10,
        )
        await heap.put(overflow)

        # Needle must still be retrievable
        assert await heap.get(needle_id) is not None
        assert await heap.size() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
