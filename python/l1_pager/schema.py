"""
L1-Pager: Core schema for the Context Garbage Collector.

Design invariant: every type here must be JSON-serializable so the
Python runtime and the TypeScript runtime can share the same Redis heap
without a translation layer.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_POINTER_PREFIX = "<PAGE_FAULT_ID:"
_POINTER_RE = re.compile(r"<PAGE_FAULT_ID:\s*([a-f0-9]+)\s*\|\s*SUMMARY:\s*(.+?)>$")

# ---------------------------------------------------------------------------
# Eviction policy
# ---------------------------------------------------------------------------


class EvictionPolicy(str, Enum):
    LRU = "lru"
    IMPORTANCE = "importance"
    # Weighted blend: importance_score - (turn_age * decay_weight)
    HYBRID = "hybrid"


@dataclass(frozen=True)
class EvictionConfig:
    """All knobs for the eviction engine. Immutable after construction."""

    # Mark criteria
    min_tokens: int = 500
    min_turns_old: int = 3

    # Sweep behaviour
    policy: EvictionPolicy = EvictionPolicy.LRU
    hybrid_decay_weight: float = 0.01

    # Heap limits
    max_heap_entries: int = 2_000
    entry_ttl_seconds: float = 3_600.0

    # Demand-paging guard
    max_page_fault_depth: int = 3

    def to_dict(self) -> dict:
        return {
            "min_tokens": self.min_tokens,
            "min_turns_old": self.min_turns_old,
            "policy": self.policy.value,
            "hybrid_decay_weight": self.hybrid_decay_weight,
            "max_heap_entries": self.max_heap_entries,
            "entry_ttl_seconds": self.entry_ttl_seconds,
            "max_page_fault_depth": self.max_page_fault_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvictionConfig:
        d = dict(d)
        d["policy"] = EvictionPolicy(d["policy"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Heap entry — the raw content stored after eviction
# ---------------------------------------------------------------------------


@dataclass
class PageHeapEntry:
    """
    Stored in Redis/InMemory after a message is swept.

    `raw_content` is the original message.content value. It must be
    JSON-serializable; complex content blocks (list[dict]) satisfy this
    by construction in LangChain.
    """

    page_id: str                    # 16-char hex prefix of SHA-256
    original_message_id: str        # LangGraph message id (for re-injection)
    original_type: str              # "ToolMessage" | "AIMessage" | …
    raw_content: Any                # str or list[dict] — the actual payload
    tool_name: str | None           # Non-None for ToolMessage
    tool_call_id: str | None        # Non-None for ToolMessage
    token_count: int
    evicted_at_turn: int
    evicted_at_ts: float = field(default_factory=time.time)
    last_accessed_ts: float | None = None
    access_count: int = 0

    # ---- serialisation ----

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "original_message_id": self.original_message_id,
            "original_type": self.original_type,
            "raw_content": self.raw_content,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "token_count": self.token_count,
            "evicted_at_turn": self.evicted_at_turn,
            "evicted_at_ts": self.evicted_at_ts,
            "last_accessed_ts": self.last_accessed_ts,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PageHeapEntry:
        return cls(**d)

    def touch(self) -> None:
        """Update LRU metadata on access."""
        self.last_accessed_ts = time.time()
        self.access_count += 1

    @property
    def lru_ts(self) -> float:
        return self.last_accessed_ts or self.evicted_at_ts


# ---------------------------------------------------------------------------
# Mark candidate — output of the Mark phase
# ---------------------------------------------------------------------------


@dataclass
class MarkCandidate:
    """A message flagged by the Mark phase as eligible for eviction."""

    message: BaseMessage
    message_index: int       # Position in the current messages list
    token_count: int
    turn_age: int            # Number of full turns since this message
    importance_score: float  # [0.0, 1.0] — lower = evict sooner
    eviction_reason: str


# ---------------------------------------------------------------------------
# Eviction record — audit log entry
# ---------------------------------------------------------------------------


@dataclass
class EvictionRecord:
    page_id: str
    original_message_id: str
    original_type: str
    token_count: int
    evicted_at_turn: int
    evicted_at_ts: float = field(default_factory=time.time)
    policy_used: str = "lru"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pointer helpers
# ---------------------------------------------------------------------------


def compute_page_id(message: BaseMessage) -> str:
    """
    Deterministic, stable 16-char hex page ID.

    Keyed on (message.id, type, content) so the same logical message
    always maps to the same heap slot — safe to call multiple times.
    """
    payload = json.dumps(
        {
            "id": message.id,
            "type": type(message).__name__,
            "content": message.content,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_pointer(page_id: str, summary: str) -> str:
    return f"<PAGE_FAULT_ID: {page_id} | SUMMARY: {summary}>"


def parse_pointer(content: Any) -> tuple[str, str] | None:
    """Return (page_id, summary) if `content` is a pointer string, else None."""
    if not isinstance(content, str):
        return None
    m = _POINTER_RE.match(content.strip())
    return (m.group(1), m.group(2)) if m else None


def is_pointer_content(content: Any) -> bool:
    return parse_pointer(content) is not None


def make_page_pointer_message(
    original: BaseMessage,
    page_id: str,
    summary: str,
    evicted_at_turn: int,
) -> BaseMessage:
    """
    Build a pointer message that replaces `original` in the message array.

    Critically, the returned message carries the *same id* as `original`
    so LangGraph's add_messages reducer performs an in-place update rather
    than appending a duplicate.
    """
    pointer_content = render_pointer(page_id, summary)
    extra_kwargs: dict = {
        "l1_page_id": page_id,
        "l1_original_type": type(original).__name__,
        "l1_evicted_at_turn": evicted_at_turn,
    }
    original_extra: dict = getattr(original, "additional_kwargs", {}) or {}

    if isinstance(original, ToolMessage):
        return ToolMessage(
            id=original.id,
            content=pointer_content,
            tool_call_id=original.tool_call_id,
            name=original.name,
            additional_kwargs={**original_extra, **extra_kwargs},
        )

    if isinstance(original, AIMessage):
        return AIMessage(
            id=original.id,
            content=pointer_content,
            additional_kwargs={**original_extra, **extra_kwargs},
        )

    # Fallback: reconstruct same type with updated content
    try:
        return type(original)(
            id=original.id,
            content=pointer_content,
            additional_kwargs={**original_extra, **extra_kwargs},
        )
    except Exception:
        return AIMessage(
            id=original.id,
            content=pointer_content,
            additional_kwargs={**original_extra, **extra_kwargs},
        )


# ---------------------------------------------------------------------------
# L1PagerState — optional TypedDict mixin for full observability
# ---------------------------------------------------------------------------


class L1PagerState(TypedDict, total=False):
    """
    State mixin for graphs that want L1Pager to be first-class in state.

    Usage::

        class MyState(L1PagerState):
            messages: Annotated[list[BaseMessage], add_messages]
            my_field: str

    If you do NOT include this mixin, L1Pager still works — it stores
    metadata in the external heap and a thread-local cache.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    l1_turn_count: int
    # page_id -> PageHeapEntry.to_dict() — survives graph checkpointing
    l1_heap_index: dict[str, dict]
    l1_eviction_log: list[dict]
    # Guards recursive demand-paging from infinite loops
    l1_page_fault_depth: int
