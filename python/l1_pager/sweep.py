"""
L1-Pager: Sweep Phase

Takes the list of MarkCandidates from the Mark phase, persists raw content
to the heap, and returns pointer messages that replace the originals in the
message array (same id → add_messages performs in-place update).

The caller receives a SweepResult with:
  - updated_messages: full message list, originals replaced by pointers
  - eviction_records: audit log for this sweep pass
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from .heap import AbstractHeap
from .mark import MarkPhase
from .schema import (
    EvictionConfig,
    EvictionRecord,
    MarkCandidate,
    PageHeapEntry,
    compute_page_id,
    make_page_pointer_message,
)
from .summarizer import generate_summary


@dataclass
class SweepResult:
    """Output of a single sweep pass."""

    # Full message list: pointer messages replace evicted originals (same id).
    # Pass this directly to add_messages; it will perform in-place updates.
    updated_messages: list[BaseMessage]
    eviction_records: list[EvictionRecord]
    tokens_freed: int
    pages_written: int

    @property
    def did_evict(self) -> bool:
        return self.pages_written > 0


class SweepPhase:
    """
    Orchestrates the full Mark→Heap→Pointer pipeline.

    Usage::

        sweep = SweepPhase(config, heap)
        result = await sweep.run(messages, current_turn=state.get("l1_turn_count", 0))
    """

    def __init__(self, config: EvictionConfig, heap: AbstractHeap) -> None:
        self._config = config
        self._heap = heap
        self._mark = MarkPhase(config)

    async def run(
        self,
        messages: list[BaseMessage],
        current_turn: int = 0,
    ) -> SweepResult:
        """
        Run the sweep pass.

        Steps:
          1. Mark: identify eviction candidates (O(n), synchronous)
          2. Heap writes: persist raw content (async, concurrent)
          3. Pointer replacement: rebuild message list

        All heap writes are issued concurrently to stay within the 15ms budget.
        """
        candidates = self._mark.scan(messages, current_turn)
        if not candidates:
            return SweepResult(
                updated_messages=list(messages),
                eviction_records=[],
                tokens_freed=0,
                pages_written=0,
            )

        # Build heap entries for all candidates before any async work
        entries = [self._build_entry(c, current_turn) for c in candidates]

        # Concurrent heap writes
        await asyncio.gather(*[self._heap.put(e) for e in entries])

        # Build the pointer-replaced message list
        evict_index = {c.message_index: (c, e) for c, e in zip(candidates, entries)}
        updated: list[BaseMessage] = []
        for idx, msg in enumerate(messages):
            if idx in evict_index:
                candidate, entry = evict_index[idx]
                summary = generate_summary(msg)
                pointer = make_page_pointer_message(msg, entry.page_id, summary, current_turn)
                updated.append(pointer)
            else:
                updated.append(msg)

        records = [
            EvictionRecord(
                page_id=e.page_id,
                original_message_id=e.original_message_id,
                original_type=e.original_type,
                token_count=e.token_count,
                evicted_at_turn=current_turn,
                policy_used=self._config.policy.value,
            )
            for e in entries
        ]

        return SweepResult(
            updated_messages=updated,
            eviction_records=records,
            tokens_freed=sum(c.token_count for c in candidates),
            pages_written=len(entries),
        )

    # ---- private helpers --------------------------------------------------

    @staticmethod
    def _build_entry(candidate: MarkCandidate, current_turn: int) -> PageHeapEntry:
        msg = candidate.message
        return PageHeapEntry(
            page_id=compute_page_id(msg),
            original_message_id=msg.id or "",
            original_type=type(msg).__name__,
            raw_content=msg.content,
            tool_name=getattr(msg, "name", None),
            tool_call_id=getattr(msg, "tool_call_id", None),
            token_count=candidate.token_count,
            evicted_at_turn=current_turn,
        )
