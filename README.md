# L1-Pager — Virtual Memory for LLM Context Windows

> **Your agent never forgets. It just pages.**

L1-Pager is a drop-in context garbage collector (CGC) for LangGraph agents. It watches your conversation, evicts large old messages to a heap, and restores them on demand — exactly like OS virtual memory, but for LLM context windows.

```
Without L1-Pager                   With L1-Pager
─────────────────                  ──────────────────────────────
┌──────────────────┐               ┌──────────────────────────────┐
│ System prompt    │               │ System prompt                │
│ Tool result 3KB  │ ← fills fast  │ [ptr: tool result evicted]   │ ← 12 chars
│ Tool result 2KB  │               │ [ptr: tool result evicted]   │
│ Tool result 4KB  │               │ Tool result 4KB              │ ← recent, kept
│ ...12 more turns │               │ ...12 more turns             │
│ Recency bias     │ ✗             │ Active window stays tight    │ ✓
└──────────────────┘               └──────────────────────────────┘
```

When the model needs an evicted value, it calls `fetch_evicted_page(page_id=...)` and gets the raw content back — zero data loss, transparent to the rest of your graph.

---

## How it works

```
Turn N                     Turn N+3                  Turn N+6
──────────────             ──────────────             ──────────────
Tool result    →  Mark  →  [ptr: abc123…]  →  Model sees ptr
(3 000 tokens)    phase    stored in heap     calls fetch_evicted_page
                                               ↓
                                              Raw content re-injected
                                              Model answers with exact value
```

1. **Mark** — O(n) scan each turn. Candidates: `> 500 tokens` AND `> 3 turns old`.
2. **Sweep** — concurrent heap writes. Pointer replaces original message in-place (same `id` → LangGraph `add_messages` deduplication works correctly).
3. **Page fault** — model calls `fetch_evicted_page`, interceptor handles it transparently before returning to your graph.

Overhead: **p99 < 1ms** on 400-message arrays (benchmarked, not estimated).

---

## Install

**Python**
```bash
pip install l1-pager

# Optional: Redis heap backend
pip install "l1-pager[redis]"
```

**TypeScript / JavaScript**
```bash
npm install l1-pager-core
```

---

## Quickstart

### Python — drop-in `create_react_agent` replacement

```python
from langchain_anthropic import ChatAnthropic
from l1_pager import EvictionConfig, build_heap
from l1_pager import create_l1_pager_react_agent

model  = ChatAnthropic(model="claude-sonnet-4-6")
config = EvictionConfig(min_tokens=500, min_turns_old=3)
heap   = build_heap(config)

# Exact same API as LangGraph's create_react_agent
agent = create_l1_pager_react_agent(model, tools=[...], heap=heap, config=config)

result = await agent.ainvoke({"messages": [HumanMessage(content="...")]})
```

### Python — wrap your existing model

```python
from l1_pager import L1Pager, EvictionConfig, build_heap

config = EvictionConfig()
heap   = build_heap(config)
pager  = L1Pager(model, heap, config)

# Use pager.ainvoke() anywhere you'd call model.invoke()
response = await pager.ainvoke(messages, current_turn=turn_count)
```

### TypeScript — LangGraph node

```typescript
import { createL1PagerReactAgent } from "@l1-pager/core";

const agent = createL1PagerReactAgent(model, tools);

const result = await agent.invoke({
  messages: [new HumanMessage("...")],
});
```

### TypeScript — lower-level

```typescript
import { L1Pager, buildHeap, DEFAULT_EVICTION_CONFIG } from "@l1-pager/core";

const heap  = buildHeap(DEFAULT_EVICTION_CONFIG);
const pager = new L1Pager(model, heap, DEFAULT_EVICTION_CONFIG);

const { aiMessage, pageFaults, tokensFreed } = await pager.ainvoke(messages, turn);
```

---

## Configuration

```python
from l1_pager import EvictionConfig, EvictionPolicy

config = EvictionConfig(
    min_tokens      = 500,        # don't evict messages smaller than this
    min_turns_old   = 3,          # don't evict messages newer than this
    max_heap_entries = 128,       # LRU cap on in-memory heap
    entry_ttl_seconds = 3_600,   # evicted pages expire after 1 hour
    max_page_fault_depth = 3,    # max re-fetch loops per model call
    policy = EvictionPolicy.LRU,  # LRU | IMPORTANCE | HYBRID
)
```

```typescript
import { EvictionConfig } from "@l1-pager/core";

const config: EvictionConfig = {
  minTokens: 500,
  minTurnsOld: 3,
  maxHeapEntries: 128,
  entryTtlSeconds: 3600,
  maxPageFaultDepth: 3,
  policy: "lru",
};
```

---

## Heap backends

### In-memory (default)

```python
heap = build_heap(config, backend="memory")
```

Zero dependencies, LRU eviction, lives in process memory.

### Redis (persistent, shared across workers)

```python
heap = await build_heap_async(config, backend="redis", redis_url="redis://localhost:6379")
```

```typescript
import { buildHeapAsync } from "@l1-pager/core";

const heap = await buildHeapAsync(config, "redis", { redisUrl: "redis://localhost:6379" });
```

Each evicted page is stored as a Redis hash with a TTL. Access resets the TTL (LRU behaviour without a sorted set).

---

## State schema

L1-Pager tracks its own counters as first-class graph state. These accumulate across turns and are accessible for observability:

| Field | Type | Description |
|---|---|---|
| `l1_turn_count` | `int` | Turns processed |
| `l1_tokens_freed_total` | `int` | Estimated tokens freed across all sweeps |
| `l1_page_fault_total` | `int` | Total page faults resolved |
| `l1_eviction_log` | `list[EvictionRecord]` | Full audit trail of every eviction |

---

## System prompt guidance

For best results, add this block to your system prompt so the model understands the compression format:

```
This conversation uses L1-Pager context compression.
Compressed messages appear as:  <PAGE_FAULT_ID: {hex} | SUMMARY: {text}>

If a question asks for SPECIFIC DATA inside a compressed reference,
call fetch_evicted_page(page_id="{hex}") to retrieve it first.
Do NOT guess — always fetch before answering precise values.
```

---

## Benchmarks

Run against synthetic conversations on Apple M-series (no Redis):

| Messages | p50 | p95 | p99 |
|---|---|---|---|
| 52 | 0.04ms | 0.07ms | 0.09ms |
| 102 | 0.08ms | 0.13ms | 0.17ms |
| 202 | 0.18ms | 0.26ms | 0.32ms |
| 402 | 0.38ms | 0.49ms | 0.55ms |

SLA target: **15ms**. Actual worst-case p99: **0.55ms** (27× headroom).

---

## Project structure

```
l1-pager/
├── python/                    # PyPI package: l1-pager
│   ├── l1_pager/
│   │   ├── schema.py          # types, pointer helpers
│   │   ├── mark.py            # O(n) mark phase
│   │   ├── heap.py            # InMemoryHeap + RedisHeap
│   │   ├── summarizer.py      # extractive one-line summaries
│   │   ├── sweep.py           # concurrent eviction
│   │   ├── tools.py           # fetch_evicted_page tool schema
│   │   ├── interceptor.py     # L1Pager — the main class
│   │   ├── langgraph_integration.py
│   │   └── wrap.py            # @l1_pager_hook decorator
│   └── tests/
└── typescript/                # npm package: @l1-pager/core
    └── src/
        ├── schema.ts
        ├── mark.ts
        ├── heap.ts
        ├── redis_heap.ts
        ├── summarizer.ts
        ├── sweep.ts
        ├── tools.ts
        ├── interceptor.ts
        ├── langgraph_node.ts
        └── index.ts
```

---

## Requirements

**Python**
- Python ≥ 3.11
- `langchain-core >= 0.3`
- `langgraph >= 0.2`
- `redis >= 4.2` _(optional, for Redis heap)_

**TypeScript**
- Node.js ≥ 18
- `@langchain/core >= 0.3` _(peer dependency)_
- `@langchain/langgraph >= 0.2` _(peer dependency)_
- `redis >= 4.0` _(optional, for Redis heap)_

---

## License

MIT
