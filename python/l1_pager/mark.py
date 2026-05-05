"""
L1-Pager: Mark Phase

Single O(n) pass over the message array. Identifies messages that satisfy
BOTH eviction criteria:
  - estimated token count > config.min_tokens
  - turn age > config.min_turns_old

The pass adds <1ms on 200-message arrays (no I/O, no tokenizer calls).
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .schema import (
    EvictionConfig,
    EvictionPolicy,
    MarkCandidate,
    is_pointer_content,
)

# ---------------------------------------------------------------------------
# Token estimation — fast path, no external dependencies
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(content: Any) -> int:
    """
    Sub-microsecond token estimate: 1 token ≈ 4 chars for English.

    Handles str, list[dict] (multi-modal content blocks), and dict.
    Accuracy within ±15% of cl100k_base — sufficient for the >500 token
    threshold; the false-eviction risk from over-counting is low because
    re-injection (page fault) is lossless.
    """
    if isinstance(content, str):
        # Collapse whitespace before dividing to avoid counting whitespace tokens
        collapsed = _WHITESPACE_RE.sub(" ", content)
        return max(1, len(collapsed) // 4)

    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                # LangChain content blocks: {"type": "text", "text": "..."}
                total += estimate_tokens(block.get("text") or block.get("content") or "")
        return max(1, total)

    if isinstance(content, dict):
        return estimate_tokens(str(content))

    return 1


# ---------------------------------------------------------------------------
# Turn-age calculation
# ---------------------------------------------------------------------------


def compute_turn_ages(messages: Sequence[BaseMessage]) -> list[int]:
    """
    Return a per-message list of turn ages (0 = current turn).

    A "turn" increments every time we see a HumanMessage after the first.
    This is more accurate than position/2 because tool-call chains can
    produce many consecutive AIMessage/ToolMessage pairs within one turn.

    Example for [H, AI, T, AI, H, AI, T]:
      turn ages: [1, 1, 1, 1, 0, 0, 0]
    """
    # Forward pass: assign a turn index to each message
    turn_index: list[int] = []
    current = 0
    for i, msg in enumerate(messages):
        if i > 0 and isinstance(msg, HumanMessage):
            current += 1
        turn_index.append(current)

    max_turn = current
    return [max_turn - t for t in turn_index]


# ---------------------------------------------------------------------------
# Ephemeral classification
# ---------------------------------------------------------------------------

_ERROR_KWS = frozenset({"error", "exception", "traceback", "failed", "panic", "fatal"})
_JSON_START = frozenset({"{", "["})


def is_ephemeral(message: BaseMessage) -> bool:
    """
    True if the message is structurally ephemeral (candidate for eviction).

    Non-evictable:
      - HumanMessage  — user intent; must never be lost
      - SystemMessage — graph invariants
      - AIMessage that is *only* a tool dispatcher (no content, has tool_calls)
      - Any message already holding a pointer (already swept)
    """
    if isinstance(message, (HumanMessage, SystemMessage)):
        return False

    content = message.content

    # Already a pointer → already swept, skip
    if is_pointer_content(content):
        return False

    # Tool outputs are the primary eviction target
    if isinstance(message, ToolMessage):
        return True

    if isinstance(message, AIMessage):
        # Pure dispatcher: no content, but has tool_calls — structural, keep it
        if not content and getattr(message, "tool_calls", None):
            return False
        # AI messages with real content are intermediate scratchpads
        return bool(content)

    # Unknown subtypes: conservative default
    return False


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def importance_score(message: BaseMessage, turn_age: int) -> float:
    """
    Heuristic importance in [0.0, 1.0]. Lower = evict sooner.

    Signals (order of magnitude):
    +0.3  error/exception keywords (debugging context is hard to regenerate)
    -0.2  pure JSON/list dump (raw data, reconstructible on demand)
    -0.1  contains large numeric sequences (metrics, log timestamps)
    -0.05 per turn of age, capped at -0.5

    Design note: these weights are intentionally coarse. A future
    IMPORTANCE policy will replace this with an embedding-based scorer.
    """
    score = 0.7  # baseline: most content is moderately important

    content_str: str
    if isinstance(message.content, str):
        content_str = message.content
    else:
        content_str = str(message.content)

    lowered = content_str.lower()

    # Error context — premium retention value
    if any(kw in lowered for kw in _ERROR_KWS):
        score += 0.3

    # Raw data blobs — low retention value
    stripped = content_str.lstrip()
    if stripped and stripped[0] in _JSON_START:
        score -= 0.2

    # Dense numeric sequences (log files, metrics exports)
    if re.search(r"\d{5,}", content_str):
        score -= 0.1

    # Recency decay
    score -= min(0.5, turn_age * 0.05)

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# MarkPhase
# ---------------------------------------------------------------------------


class MarkPhase:
    """
    Implements the Mark pass of the Context GC.

    Usage::

        mark = MarkPhase(config)
        candidates = mark.scan(state["messages"], state.get("l1_turn_count", 0))
        # candidates is sorted: highest eviction priority first
    """

    __slots__ = ("config",)

    def __init__(self, config: EvictionConfig) -> None:
        self.config = config

    def scan(
        self,
        messages: Sequence[BaseMessage],
        current_turn: int = 0,
    ) -> list[MarkCandidate]:
        """
        O(n) scan. Returns candidates sorted by eviction priority.

        Both criteria must be satisfied:
          token_count > min_tokens   AND   turn_age > min_turns_old
        """
        ages = compute_turn_ages(messages)
        cfg = self.config
        candidates: list[MarkCandidate] = []

        for idx, msg in enumerate(messages):
            if not is_ephemeral(msg):
                continue

            tokens = estimate_tokens(msg.content)
            if tokens <= cfg.min_tokens:
                continue

            age = ages[idx]
            if age <= cfg.min_turns_old:
                continue

            imp = importance_score(msg, age)
            candidates.append(
                MarkCandidate(
                    message=msg,
                    message_index=idx,
                    token_count=tokens,
                    turn_age=age,
                    importance_score=imp,
                    eviction_reason=self._reason(msg, tokens, age),
                )
            )

        self._sort(candidates)
        return candidates

    # ---- private helpers --------------------------------------------------

    def _sort(self, candidates: list[MarkCandidate]) -> None:
        policy = self.config.policy
        w = self.config.hybrid_decay_weight

        if policy == EvictionPolicy.LRU:
            # Oldest first
            candidates.sort(key=lambda c: -c.turn_age)
        elif policy == EvictionPolicy.IMPORTANCE:
            # Least important first
            candidates.sort(key=lambda c: c.importance_score)
        else:  # HYBRID
            # Low importance + old age both push toward eviction
            candidates.sort(
                key=lambda c: c.importance_score - (c.turn_age * w)
            )

    @staticmethod
    def _reason(msg: BaseMessage, tokens: int, age: int) -> str:
        return f"{type(msg).__name__} | ~{tokens} tokens | {age} turns old"
