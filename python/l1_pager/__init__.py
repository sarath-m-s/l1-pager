"""L1-Pager: Context Garbage Collector for LangGraph agents."""
from .schema import (
    EvictionConfig,
    EvictionPolicy,
    EvictionRecord,
    L1PagerState,
    MarkCandidate,
    PageHeapEntry,
    compute_page_id,
    is_pointer_content,
    make_page_pointer_message,
    parse_pointer,
    render_pointer,
)
from .mark import MarkPhase, estimate_tokens
from .heap import AbstractHeap, InMemoryHeap, RedisHeap, build_heap
from .summarizer import generate_summary
from .sweep import SweepPhase, SweepResult
from .tools import fetch_evicted_page_tool, is_page_fault_tool_call, FETCH_EVICTED_PAGE_NAME
from .interceptor import L1Pager, L1PagerResponse
from .langgraph_integration import (
    L1PagerGraphState,
    create_l1_pager_node,
    create_l1_pager_react_agent,
    should_continue,
)
from .wrap import WrapModelCallMiddleware, l1_pager_hook, wrap_model_call

__all__ = [
    # Main class
    "L1Pager",
    "L1PagerResponse",
    # LangGraph integration
    "L1PagerGraphState",
    "create_l1_pager_node",
    "create_l1_pager_react_agent",
    "should_continue",
    # wrap_model_call
    "WrapModelCallMiddleware",
    "l1_pager_hook",
    "wrap_model_call",
    # Schema
    "EvictionConfig",
    "EvictionPolicy",
    "EvictionRecord",
    "L1PagerState",
    "MarkCandidate",
    "PageHeapEntry",
    "compute_page_id",
    "is_pointer_content",
    "make_page_pointer_message",
    "parse_pointer",
    "render_pointer",
    # Mark
    "MarkPhase",
    "estimate_tokens",
    # Heap
    "AbstractHeap",
    "InMemoryHeap",
    "RedisHeap",
    "build_heap",
    # Summarizer
    "generate_summary",
    # Sweep
    "SweepPhase",
    "SweepResult",
    # Tools
    "fetch_evicted_page_tool",
    "is_page_fault_tool_call",
    "FETCH_EVICTED_PAGE_NAME",
]
