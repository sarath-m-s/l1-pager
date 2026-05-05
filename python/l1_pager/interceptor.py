"""
L1-Pager: The Interceptor — L1Pager class.

Drop-in replacement for any LangChain BaseChatModel. Wraps the model with:
  1. Pre-call sweep: evicts old large messages, replaces with pointers
  2. Post-call page-fault detection: if the model calls fetch_evicted_page,
     retrieve raw content from heap, re-inject, re-invoke (demand paging)
  3. Recursive depth guard: prevents infinite page-fault loops

Usage::

    from l1_pager import L1Pager, EvictionConfig, build_heap

    model = ChatOpenAI(model="gpt-4o")
    heap = build_heap(EvictionConfig(), backend="memory")
    pager = L1Pager(model=model, heap=heap)

    # In your LangGraph node:
    async def agent_node(state):
        result = await pager.ainvoke(state["messages"])
        # result.messages contains pointer-replaced messages + AI response
        return {"messages": result.messages, "l1_eviction_log": result.eviction_records}

    # Or use the model-like interface for create_react_agent:
    agent = create_react_agent(pager.as_model(), pager.tools)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from .heap import AbstractHeap, build_heap
from .schema import EvictionConfig, EvictionRecord, PageHeapEntry
from .sweep import SweepPhase, SweepResult
from .tools import FETCH_EVICTED_PAGE_NAME, fetch_evicted_page_tool, is_page_fault_tool_call

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


@dataclass
class L1PagerResponse:
    """
    Returned by L1Pager.ainvoke().

    `messages` should be passed directly to `add_messages` in your node
    return dict. It contains:
      - pointer replacements for evicted messages (same IDs → in-place update)
      - the model's final AIMessage appended at the end
    """

    ai_message: AIMessage
    # Pointer replacements + final AI message — pass to {"messages": result.messages}
    messages: list[BaseMessage]
    eviction_records: list[EvictionRecord]
    page_faults: int
    tokens_freed: int


# ---------------------------------------------------------------------------
# L1Pager
# ---------------------------------------------------------------------------


class L1Pager:
    """
    Context Garbage Collector for LangGraph agents.

    Thread-safe: each ainvoke() call is independent (no shared mutable state
    outside the heap, which uses its own asyncio.Lock).
    """

    def __init__(
        self,
        model: BaseChatModel,
        heap: AbstractHeap | None = None,
        config: EvictionConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or EvictionConfig()
        self._heap = heap or build_heap(self._config, backend="memory")
        self._sweep = SweepPhase(self._config, self._heap)

        # Bind fetch_evicted_page to the model so it appears in the tool schema
        self._bound_model = model.bind_tools([fetch_evicted_page_tool])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Expose the page-fault tool for use with create_react_agent."""
        return [fetch_evicted_page_tool]

    def as_model(self) -> _ModelShim:
        """
        Return a shim that presents L1Pager as a BaseChatModel-compatible
        object for use with create_react_agent or similar helpers.
        """
        return _ModelShim(self)

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        current_turn: int = 0,
        **kwargs: Any,
    ) -> L1PagerResponse:
        """
        Full pipeline: sweep → model call → page-fault recovery.

        Args:
            messages: The current message array from graph state.
            current_turn: Optional turn counter (pass state["l1_turn_count"]).
            **kwargs: Forwarded to the underlying model's ainvoke().
        """
        # ---- 1. Sweep -------------------------------------------------
        sweep_result = await self._sweep.run(messages, current_turn)

        # ---- 2. Model call --------------------------------------------
        swept = sweep_result.updated_messages
        ai_response = await self._bound_model.ainvoke(swept, **kwargs)

        # ---- 3. Page-fault recovery loop ------------------------------
        total_faults = 0
        current_messages = swept

        while self._has_page_fault(ai_response):
            if total_faults >= self._config.max_page_fault_depth:
                log.warning(
                    "L1Pager: max_page_fault_depth (%d) reached, stopping recovery",
                    self._config.max_page_fault_depth,
                )
                break

            ai_response, current_messages = await self._recover_page_fault(
                current_messages, ai_response
            )
            total_faults += 1

        # ---- 4. Assemble result ---------------------------------------
        # Pointer replacements (have original IDs → in-place update via add_messages)
        # followed by the final AI response (new ID → appended)
        output_messages = sweep_result.updated_messages + [ai_response]

        return L1PagerResponse(
            ai_message=ai_response,
            messages=output_messages,
            eviction_records=sweep_result.eviction_records,
            page_faults=total_faults,
            tokens_freed=sweep_result.tokens_freed,
        )

    def invoke(self, messages: list[BaseMessage], current_turn: int = 0, **kwargs: Any) -> L1PagerResponse:
        """Synchronous wrapper — runs the async pipeline in a new event loop."""
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, current_turn=current_turn, **kwargs)
        )

    # ------------------------------------------------------------------
    # Page-fault recovery
    # ------------------------------------------------------------------

    @staticmethod
    def _has_page_fault(response: AIMessage) -> bool:
        tool_calls = getattr(response, "tool_calls", None) or []
        return any(is_page_fault_tool_call(tc) for tc in tool_calls)

    async def _recover_page_fault(
        self,
        messages: list[BaseMessage],
        ai_response: AIMessage,
    ) -> tuple[AIMessage, list[BaseMessage]]:
        """
        Handle all fetch_evicted_page calls in `ai_response`.

        For each call:
          a) Halt — do NOT pass to ToolNode
          b) Retrieve raw content from heap
          c) Build a ToolMessage with re-injected content
          d) Append [ai_response, ...tool_messages] to message array
          e) Re-invoke the model

        Returns (new_ai_response, updated_message_array).
        """
        tool_calls = getattr(ai_response, "tool_calls", []) or []
        page_fault_calls = [tc for tc in tool_calls if is_page_fault_tool_call(tc)]

        # Fetch all pages concurrently
        async def _fetch(tc: dict) -> ToolMessage:
            page_id: str = tc["args"].get("page_id", "")
            entry: PageHeapEntry | None = await self._heap.get(page_id)

            if entry is None:
                content = (
                    f"[L1Pager: page '{page_id}' not found — "
                    f"it may have expired (TTL={self._config.entry_ttl_seconds}s) "
                    f"or the page_id is incorrect.]"
                )
            else:
                content = entry.raw_content
                log.debug(
                    "L1Pager: page fault resolved for %s (tool=%s, tokens=%d)",
                    page_id,
                    entry.tool_name,
                    entry.token_count,
                )

            return ToolMessage(
                content=content,
                tool_call_id=tc["id"],
                name=FETCH_EVICTED_PAGE_NAME,
            )

        tool_messages = await asyncio.gather(*[_fetch(tc) for tc in page_fault_calls])

        # Extend the message array and re-invoke
        extended = list(messages) + [ai_response] + list(tool_messages)
        new_response = await self._bound_model.ainvoke(extended)
        return new_response, extended


# ---------------------------------------------------------------------------
# _ModelShim — thin adapter so L1Pager works with create_react_agent
# ---------------------------------------------------------------------------


class _ModelShim:
    """
    Presents L1Pager as a BaseChatModel-compatible callable for use with
    ``create_react_agent`` and other LangGraph helpers.

    ``bind_tools(user_tools)`` is called by create_react_agent internally.
    We must forward user tools to the underlying model while ensuring
    ``fetch_evicted_page`` is also present in the schema — otherwise the LLM
    cannot call it to trigger a page fault.
    """

    def __init__(self, pager: L1Pager) -> None:
        self._pager = pager

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        result = await self._pager.ainvoke(messages, **kwargs)
        return result.ai_message

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        result = self._pager.invoke(messages, **kwargs)
        return result.ai_message

    def bind_tools(self, tools: list, **kwargs: Any) -> "_ModelShim":
        """
        Bind user tools AND fetch_evicted_page to the underlying model.

        Deduplicates: if fetch_evicted_page is already in the list
        (e.g. user passed it explicitly) we don't add it twice.
        """
        tool_names = {getattr(t, "name", None) for t in tools}
        all_tools = list(tools)
        if FETCH_EVICTED_PAGE_NAME not in tool_names:
            all_tools.append(fetch_evicted_page_tool)

        # Rebind the underlying model with the complete tool list
        new_bound = self._pager._model.bind_tools(all_tools, **kwargs)

        # Clone the pager with the updated bound model
        clone = L1Pager.__new__(L1Pager)
        clone._model = self._pager._model
        clone._config = self._pager._config
        clone._heap = self._pager._heap
        clone._sweep = self._pager._sweep
        clone._bound_model = new_bound
        return _ModelShim(clone)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pager._bound_model, name)
