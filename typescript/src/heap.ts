/**
 * L1-Pager: Heap interface and implementations (TypeScript)
 *
 * InMemoryHeap — zero dependencies, ships in the main bundle.
 * RedisHeap    — lives in redis_heap.ts; imported lazily so redis is optional.
 */

import { EvictionConfig, PageHeapEntry } from "./schema.js";

// ---------------------------------------------------------------------------
// Abstract interface
// ---------------------------------------------------------------------------

export interface AbstractHeap {
  put(entry: PageHeapEntry): Promise<void>;
  get(pageId: string): Promise<PageHeapEntry | null>;
  delete(pageId: string): Promise<void>;
  size(): Promise<number>;
}

// ---------------------------------------------------------------------------
// InMemoryHeap — Map-based LRU, no external deps
// ---------------------------------------------------------------------------

export class InMemoryHeap implements AbstractHeap {
  private store = new Map<string, PageHeapEntry>();

  constructor(private readonly config: EvictionConfig) {}

  async put(entry: PageHeapEntry): Promise<void> {
    this.store.delete(entry.pageId);
    this.store.set(entry.pageId, entry);
    while (this.store.size > this.config.maxHeapEntries) {
      const oldest = this.store.keys().next().value;
      if (oldest) this.store.delete(oldest);
    }
  }

  async get(pageId: string): Promise<PageHeapEntry | null> {
    const entry = this.store.get(pageId);
    if (!entry) return null;
    if (Date.now() / 1_000 - entry.evictedAtTs > this.config.entryTtlSeconds) {
      this.store.delete(pageId);
      return null;
    }
    entry.touch();
    this.store.delete(pageId);
    this.store.set(pageId, entry);
    return entry;
  }

  async delete(pageId: string): Promise<void> {
    this.store.delete(pageId);
  }

  async size(): Promise<number> {
    return this.store.size;
  }

  allIds(): string[] {
    return [...this.store.keys()];
  }
}

// ---------------------------------------------------------------------------
// Factory — RedisHeap is imported lazily to keep redis an optional peer dep
// ---------------------------------------------------------------------------

export async function buildHeapAsync(
  config: EvictionConfig,
  backend: "memory" | "redis" = "memory",
  options?: { redisUrl?: string }
): Promise<AbstractHeap> {
  if (backend === "memory") return new InMemoryHeap(config);
  if (backend === "redis") {
    const { RedisHeap } = await import("./redis_heap.js");
    return new RedisHeap(config, options?.redisUrl);
  }
  throw new Error(`Unknown heap backend: ${backend}. Choose 'memory' or 'redis'.`);
}

/** Synchronous convenience — memory only. For Redis use buildHeapAsync(). */
export function buildHeap(
  config: EvictionConfig,
  backend: "memory" = "memory"
): InMemoryHeap {
  return new InMemoryHeap(config);
}
