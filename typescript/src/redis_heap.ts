/**
 * L1-Pager: RedisHeap (TypeScript)
 * Mirrors python/l1_pager/heap.py RedisHeap exactly.
 *
 * Key format: l1pager:heap:{page_id} → JSON string (wire format).
 * TTL enforced by Redis EXPIRE. LRU signal: accessing an entry resets its TTL.
 * size() uses SCAN — safe on large keyspaces.
 *
 * Requires the `redis` npm package (ioredis or node-redis v4+).
 * Install: npm install redis
 */

import { AbstractHeap } from "./heap.js";
import { EvictionConfig, PageHeapEntry, PageHeapEntryWire } from "./schema.js";

const KEY_PREFIX = "l1pager:heap:";

export class RedisHeap implements AbstractHeap {
  private client: any = null; // redis.RedisClientType, lazily created

  constructor(
    private readonly config: EvictionConfig,
    private readonly redisUrl: string = "redis://localhost:6379"
  ) {}

  private async conn(): Promise<any> {
    if (this.client) return this.client;
    // Dynamic import — keeps redis an optional peer dep
    let createClient: any;
    try {
      ({ createClient } = await import("redis"));
    } catch (err) {
      throw new Error(
        "RedisHeap requires the 'redis' package. Install with: npm install redis"
      );
    }
    this.client = createClient({ url: this.redisUrl });
    await this.client.connect();
    return this.client;
  }

  private key(pageId: string): string {
    return `${KEY_PREFIX}${pageId}`;
  }

  async put(entry: PageHeapEntry): Promise<void> {
    const client = await this.conn();
    const ttl = Math.floor(this.config.entryTtlSeconds);
    await client.set(this.key(entry.pageId), JSON.stringify(entry.toWire()), {
      EX: ttl,
    });
  }

  async get(pageId: string): Promise<PageHeapEntry | null> {
    const client = await this.conn();
    const raw = await client.get(this.key(pageId));
    if (!raw) return null;

    const wire: PageHeapEntryWire = JSON.parse(raw);
    const entry = PageHeapEntry.fromWire(wire);
    entry.touch();

    // Persist updated access metadata + reset TTL
    const ttl = Math.floor(this.config.entryTtlSeconds);
    await client.set(this.key(pageId), JSON.stringify(entry.toWire()), { EX: ttl });
    return entry;
  }

  async delete(pageId: string): Promise<void> {
    const client = await this.conn();
    await client.del(this.key(pageId));
  }

  async size(): Promise<number> {
    const client = await this.conn();
    let count = 0;
    for await (const _ of client.scanIterator({ MATCH: `${KEY_PREFIX}*`, COUNT: 100 })) {
      count++;
    }
    return count;
  }

  async close(): Promise<void> {
    if (this.client) {
      await this.client.quit();
      this.client = null;
    }
  }
}
