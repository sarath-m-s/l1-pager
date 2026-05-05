"""
L1-Pager: ``wrap_model_call`` adapter.

LangGraph exposes a hook pattern for intercepting model calls.  In Python
LangGraph v0.2+ the canonical way to intercept is to pass a custom callable
as the ``model`` argument to ``create_react_agent``.  This module provides:

1. ``wrap_model_call(model, pager)``
   Returns a callable with the same signature as the model's ``ainvoke``,
   but with the L1Pager sweep + page-fault pipeline injected.  Use this when
   you need to retrofit L1Pager onto an existing model instance without
   changing the graph structure.

2. ``@l1_pager_hook`` decorator
   Class-decorator for LangGraph node functions.  Wraps every ``ainvoke``
   call inside the decorated node with the full L1Pager pipeline.

3. ``WrapModelCallMiddleware``
   Callable class that satisfies the LangGraph ``wrapModelCall`` signature
   used in LangChain's ``Runnable`` API.  Can be passed directly as:
       model = WrapModelCallMiddleware(model, pager)
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from .interceptor import L1Pager, L1PagerResponse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Functional wrap
# ---------------------------------------------------------------------------


def wrap_model_call(
    model: Any,
    pager: L1Pager,
) -> Callable[[list[BaseMessage], int], Any]:
    """
    Wrap a model's ainvoke with the L1Pager pipeline.

    Returns an async callable:  ``(messages, current_turn=0, **kwargs) → AIMessage``

    This is the lowest-level hook — it gives you an async function that
    behaves exactly like ``model.ainvoke`` but with sweep + page-fault recovery.

    Usage::

        pager = L1Pager(model, heap)
        call = wrap_model_call(model, pager)

        # In your node:
        response = await call(state["messages"], current_turn=state.get("l1_turn_count", 0))
    """

    @functools.wraps(model.ainvoke if hasattr(model, "ainvoke") else model)
    async def _wrapped(
        messages: list[BaseMessage],
        current_turn: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        result: L1PagerResponse = await pager.ainvoke(
            messages, current_turn=current_turn, **kwargs
        )
        return result.ai_message

    return _wrapped


# ---------------------------------------------------------------------------
# 2. Node decorator
# ---------------------------------------------------------------------------


def l1_pager_hook(pager: L1Pager) -> Callable:
    """
    Class or function decorator that injects L1Pager into any node that calls
    a model's ``ainvoke``.

    Decorated function receives an extra keyword argument ``_pager_result``
    with the full ``L1PagerResponse`` if it declares the parameter; otherwise
    it is silently omitted.

    Usage::

        pager = L1Pager(model, heap)

        @l1_pager_hook(pager)
        async def agent_node(state):
            # model.ainvoke is transparently replaced by pager.ainvoke
            response = await model.ainvoke(state["messages"])
            return {"messages": [response]}

    Note: this decorator patches the pager's ``ainvoke`` into the node's
    closure.  It does NOT modify ``model`` in place.
    """

    def decorator(node_fn: Callable) -> Callable:
        import inspect

        sig = inspect.signature(node_fn)
        accepts_pager_result = "_pager_result" in sig.parameters

        @functools.wraps(node_fn)
        async def wrapper(state: dict, **kwargs: Any) -> dict:
            messages: list[BaseMessage] = state.get("messages", [])
            turn: int = state.get("l1_turn_count", 0)
            result: L1PagerResponse = await pager.ainvoke(messages, current_turn=turn)

            # Inject a patched state where messages are already swept
            swept_state = dict(state)
            swept_state["messages"] = result.updated_messages if hasattr(result, "updated_messages") else messages
            swept_state["_pager_result"] = result

            if accepts_pager_result:
                node_output = await node_fn(swept_state, **kwargs)
            else:
                # Remove _pager_result so the wrapped function doesn't see it
                del swept_state["_pager_result"]
                node_output = await node_fn(swept_state, **kwargs)

            # Merge eviction metadata into the node's output
            if isinstance(node_output, dict):
                node_output.setdefault("l1_turn_count", turn + 1)
                if result.eviction_records:
                    existing = node_output.get("l1_eviction_log", [])
                    node_output["l1_eviction_log"] = existing + [r.to_dict() for r in result.eviction_records]

            return node_output

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 3. WrapModelCallMiddleware — Runnable-compatible callable class
# ---------------------------------------------------------------------------


class WrapModelCallMiddleware:
    """
    Callable class that satisfies the LangGraph/LangChain ``wrapModelCall``
    pattern.  Can be used anywhere a model is expected.

    The class delegates ``bind_tools``, ``with_config``, ``with_retry`` etc.
    to the underlying pager shim, so it is compatible with pipeline builders
    that chain ``.bind_tools(...).with_config(...).ainvoke(...)``.

    Usage::

        pager = L1Pager(ChatOpenAI(), heap)
        model = WrapModelCallMiddleware(pager)

        # Works with create_react_agent:
        agent = create_react_agent(model, tools=[search])
    """

    def __init__(self, pager: L1Pager) -> None:
        self._pager = pager
        self._shim = pager.as_model()

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        current_turn = (config or {}).get("configurable", {}).get("l1_turn_count", 0)
        result = await self._pager.ainvoke(messages, current_turn=current_turn, **kwargs)
        return result.ai_message

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        return self._pager.invoke(messages, **kwargs).ai_message

    def bind_tools(self, tools: list, **kwargs: Any) -> "WrapModelCallMiddleware":
        new_shim = self._shim.bind_tools(tools, **kwargs)
        clone = WrapModelCallMiddleware.__new__(WrapModelCallMiddleware)
        clone._pager = new_shim._pager
        clone._shim = new_shim
        return clone

    def __getattr__(self, name: str) -> Any:
        return getattr(self._shim, name)
