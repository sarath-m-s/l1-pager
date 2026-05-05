"""
L1-Pager: Fast extractive summarizer for pointer generation.

Constraint: must complete in <1ms per message. No LLM call.
Produces a 1-sentence description so the model can decide whether
to request the evicted page or ignore it.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_SENTENCE_RE = re.compile(r"[.!?]\s")
_MAX_SUMMARY_CHARS = 140


def _first_sentence(text: str) -> str:
    # Collapse newlines so the summary is always a single line
    text = " ".join(text.split())
    m = _SENTENCE_RE.search(text)
    return text[: m.end()].strip() if m else text[:_MAX_SUMMARY_CHARS]


def _summarize_json(text: str) -> str:
    """Try to extract meaningful keys from a JSON blob."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            keys = list(obj.keys())[:5]
            return f"JSON object with keys: {', '.join(keys)}"
        if isinstance(obj, list):
            return f"JSON array with {len(obj)} items"
    except (json.JSONDecodeError, ValueError):
        pass
    return ""


def _char_count(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def generate_summary(message: BaseMessage) -> str:
    """
    Generate a ≤140-char, single-line extractive summary for the pointer string.

    Strategy (in order):
    1. If content starts with JSON → extract key names
    2. Extract the first sentence
    3. Fall back to a type+size template

    Invariant: the returned string contains no newline characters.
    """
    content = message.content
    content_str = content if isinstance(content, str) else str(content)
    content_str = content_str.strip()
    char_count = len(content_str)

    raw: str

    # JSON blobs
    if content_str and content_str[0] in ("{", "["):
        summary = _summarize_json(content_str)
        if summary:
            raw = summary[:_MAX_SUMMARY_CHARS]
        else:
            raw = f"{type(message).__name__} ({char_count} chars)"
    elif isinstance(message, ToolMessage) and message.name:
        first = _first_sentence(content_str)
        if len(first) > 15:
            raw = f"{message.name}: {first}"[:_MAX_SUMMARY_CHARS]
        else:
            raw = f"{message.name} output ({char_count} chars)"
    else:
        first = _first_sentence(content_str)
        raw = first[:_MAX_SUMMARY_CHARS] if len(first) > 15 else f"{type(message).__name__} ({char_count} chars)"

    # Invariant: no newlines — a multiline summary would break the pointer regex
    return " ".join(raw.split())
