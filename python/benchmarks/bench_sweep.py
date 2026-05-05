"""
L1-Pager latency benchmark.

Measures the overhead L1Pager adds *beyond* the actual model call latency.
The SLA is p99 < 15ms for all message-array sizes tested.

Run with:
    python benchmarks/bench_sweep.py

The script prints a table and exits non-zero if any p99 exceeds 15ms.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, ".")  # run from python/ directory

from l1_pager import EvictionConfig, EvictionPolicy, build_heap
from l1_pager.mark import MarkPhase
from l1_pager.sweep import SweepPhase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NEEDLE_JSON = json.dumps({
    "transaction_id": "TXN-BENCH-001",
    "amount": 9_999.99,
    "status": "settled",
    "metadata": {"rows": list(range(50))},
})
PADDING = "\n".join(f"LOG [{i:04d}] event processed" for i in range(120))
LARGE_CONTENT = NEEDLE_JSON + "\n\n" + PADDING


def build_msgs(num_turns: int) -> list:
    msgs = [SystemMessage(content="You are an assistant.", id=str(uuid4()))]
    tc_id = str(uuid4())
    msgs.append(HumanMessage(content="Check the transaction.", id=str(uuid4())))
    msgs.append(AIMessage(
        content="",
        tool_calls=[{"name": "check_txn", "args": {}, "id": tc_id}],
        id=str(uuid4()),
    ))
    msgs.append(ToolMessage(content=LARGE_CONTENT, tool_call_id=tc_id, name="check_txn", id=str(uuid4())))
    for i in range(1, num_turns):
        msgs.append(HumanMessage(content=f"Q{i}", id=str(uuid4())))
        msgs.append(AIMessage(content=f"A{i}", id=str(uuid4())))
    return msgs


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

ITERATIONS = 500
SIZES = [25, 50, 100, 200]
SLA_P99_MS = 15.0


async def bench_mark_phase(msgs: list, label: str) -> dict:
    config = EvictionConfig(min_tokens=100, min_turns_old=3)
    phase = MarkPhase(config)
    times_ms = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        phase.scan(msgs)
        times_ms.append((time.perf_counter() - t0) * 1_000)
    return {"label": label, "component": "MarkPhase.scan", "times": times_ms}


async def bench_sweep_phase(msgs: list, label: str) -> dict:
    config = EvictionConfig(min_tokens=100, min_turns_old=3)
    heap = build_heap(config, backend="memory")
    phase = SweepPhase(config, heap)
    times_ms = []
    # Warm up heap so the first few writes don't skew timings
    await phase.run(msgs, current_turn=100)
    for _ in range(ITERATIONS):
        fresh_heap = build_heap(config, backend="memory")
        fresh_phase = SweepPhase(config, fresh_heap)
        t0 = time.perf_counter()
        await fresh_phase.run(msgs, current_turn=100)
        times_ms.append((time.perf_counter() - t0) * 1_000)
    return {"label": label, "component": "SweepPhase.run", "times": times_ms}


async def bench_full_overhead(msgs: list, label: str) -> dict:
    """
    Measure the L1Pager overhead excluding actual model latency.

    We use a no-op model so the only measurable time is the sweep +
    page-fault detection logic.
    """
    from l1_pager import L1Pager
    config = EvictionConfig(min_tokens=100, min_turns_old=3)
    heap = build_heap(config, backend="memory")

    class _NoOpModel:
        def bind_tools(self, tools, **kw): return self
        async def ainvoke(self, messages, **kw):
            return AIMessage(content="ok", id=str(uuid4()))

    pager = L1Pager(model=_NoOpModel(), heap=heap, config=config)
    times_ms = []
    for _ in range(ITERATIONS):
        # Rebuild heap so sweep actually has work to do each iteration
        fresh_heap = build_heap(config, backend="memory")
        fresh_pager = L1Pager(model=_NoOpModel(), heap=fresh_heap, config=config)
        t0 = time.perf_counter()
        await fresh_pager.ainvoke(msgs, current_turn=100)
        times_ms.append((time.perf_counter() - t0) * 1_000)
    return {"label": label, "component": "L1Pager.ainvoke (sweep+detect)", "times": times_ms}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarise(result: dict) -> tuple[float, float, float, float]:
    t = result["times"]
    quantiles = statistics.quantiles(t, n=100)
    return (
        statistics.mean(t),
        statistics.median(t),
        quantiles[94],  # p95
        quantiles[98],  # p99
    )


async def main() -> int:
    print(f"\n{'L1-Pager Latency Benchmark':=^72}")
    print(f"  Iterations per cell: {ITERATIONS}")
    print(f"  SLA: p99 < {SLA_P99_MS}ms")
    print()

    header = f"{'Component':<35} {'Msgs':>5} {'Mean':>7} {'Median':>7} {'p95':>7} {'p99':>7}  {'SLA'}"
    print(header)
    print("-" * len(header))

    violations: list[str] = []

    for num_turns in SIZES:
        msgs = build_msgs(num_turns)
        label = f"{len(msgs)} msgs ({num_turns} turns)"

        for bench_fn in [bench_mark_phase, bench_sweep_phase, bench_full_overhead]:
            result = await bench_fn(msgs, label)
            mean, med, p95, p99 = summarise(result)
            sla_ok = p99 < SLA_P99_MS
            marker = "✓" if sla_ok else "✗ VIOLATION"
            print(
                f"{result['component']:<35} {len(msgs):>5} "
                f"{mean:>6.2f}ms {med:>6.2f}ms {p95:>6.2f}ms {p99:>6.2f}ms  {marker}"
            )
            if not sla_ok:
                violations.append(
                    f"{result['component']} @ {label}: p99={p99:.2f}ms > {SLA_P99_MS}ms"
                )
        print()

    print("=" * 72)

    if violations:
        print(f"\n[FAIL] {len(violations)} SLA violation(s):")
        for v in violations:
            print(f"  • {v}")
        return 1

    print("\n[PASS] All measurements within 15ms p99 SLA.")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
