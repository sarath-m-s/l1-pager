"""
L1-Pager: System tool definitions.

`fetch_evicted_page` is a sentinel tool. The L1Pager interceptor intercepts
the tool call BEFORE it reaches any ToolNode, so the `func` implementation
here is never executed in production. It exists only to generate the JSON
schema that gets bound to the model.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

FETCH_EVICTED_PAGE_NAME = "fetch_evicted_page"


class FetchEvictedPageInput(BaseModel):
    page_id: str = Field(
        description=(
            "The hexadecimal page ID from the PAGE_FAULT_ID reference. "
            "Example: if you see '<PAGE_FAULT_ID: abc123 | SUMMARY: ...>', "
            "pass 'abc123' as page_id."
        )
    )


def _fetch_evicted_page_placeholder(page_id: str) -> str:
    # This function is never called in production.
    # The L1Pager interceptor handles retrieval before tool execution.
    return f"[L1Pager internal: page {page_id} will be retrieved by the interceptor]"


fetch_evicted_page_tool = StructuredTool(
    name=FETCH_EVICTED_PAGE_NAME,
    description=(
        "Retrieve the original content of a context page that was compressed "
        "and evicted from the active context window to save space. "
        "Use this when you see a <PAGE_FAULT_ID: {id} | SUMMARY: {summary}> "
        "reference and need the full original content to answer accurately."
    ),
    args_schema=FetchEvictedPageInput,
    func=_fetch_evicted_page_placeholder,
    coroutine=None,
)


def is_page_fault_tool_call(tool_call: dict) -> bool:
    """True if a tool_call dict refers to fetch_evicted_page."""
    return tool_call.get("name") == FETCH_EVICTED_PAGE_NAME
