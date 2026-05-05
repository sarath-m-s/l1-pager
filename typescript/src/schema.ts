/**
 * L1-Pager: Core schema for the Context Garbage Collector.
 *
 * All types here are JSON-serializable so the TS runtime and the Python
 * runtime can share the same Redis heap key-space without translation.
 */

import {
  AIMessage,
  BaseMessage,
  MessageContent,
  ToolMessage,
} from "@langchain/core/messages";
import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { createHash } from "crypto";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const PAGE_POINTER_PREFIX = "<PAGE_FAULT_ID:" as const;
const POINTER_RE = /^<PAGE_FAULT_ID:\s*([a-f0-9]+)\s*\|\s*SUMMARY:\s*(.+?)>$/;

// ---------------------------------------------------------------------------
// Eviction policy
// ---------------------------------------------------------------------------

export type EvictionPolicy = "lru" | "importance" | "hybrid";

export interface EvictionConfig {
  /** Minimum estimated tokens for a message to be eviction-eligible. */
  readonly minTokens: number;
  /** Minimum turns-old a message must be to be eviction-eligible. */
  readonly minTurnsOld: number;
  readonly policy: EvictionPolicy;
  /** Weight applied to turn age in the hybrid policy. */
  readonly hybridDecayWeight: number;
  readonly maxHeapEntries: number;
  readonly entryTtlSeconds: number;
  /** Guard: maximum recursive depth for demand-paging page faults. */
  readonly maxPageFaultDepth: number;
}

export const DEFAULT_EVICTION_CONFIG: EvictionConfig = {
  minTokens: 500,
  minTurnsOld: 3,
  policy: "lru",
  hybridDecayWeight: 0.01,
  maxHeapEntries: 2_000,
  entryTtlSeconds: 3_600,
  maxPageFaultDepth: 3,
};

// Wire Python <-> TS: field names use snake_case in the wire format
export function evictionConfigToWire(cfg: EvictionConfig): Record<string, unknown> {
  return {
    min_tokens: cfg.minTokens,
    min_turns_old: cfg.minTurnsOld,
    policy: cfg.policy,
    hybrid_decay_weight: cfg.hybridDecayWeight,
    max_heap_entries: cfg.maxHeapEntries,
    entry_ttl_seconds: cfg.entryTtlSeconds,
    max_page_fault_depth: cfg.maxPageFaultDepth,
  };
}

export function evictionConfigFromWire(raw: Record<string, unknown>): EvictionConfig {
  return {
    minTokens: raw["min_tokens"] as number,
    minTurnsOld: raw["min_turns_old"] as number,
    policy: raw["policy"] as EvictionPolicy,
    hybridDecayWeight: raw["hybrid_decay_weight"] as number,
    maxHeapEntries: raw["max_heap_entries"] as number,
    entryTtlSeconds: raw["entry_ttl_seconds"] as number,
    maxPageFaultDepth: raw["max_page_fault_depth"] as number,
  };
}

// ---------------------------------------------------------------------------
// PageHeapEntry — raw content stored in the heap after eviction
// ---------------------------------------------------------------------------

/**
 * Wire format (snake_case) so both Python and TS agents read/write
 * the same Redis hash without a translation step.
 */
export interface PageHeapEntryWire {
  page_id: string;
  original_message_id: string;
  original_type: string;
  raw_content: MessageContent;
  tool_name: string | null;
  tool_call_id: string | null;
  token_count: number;
  evicted_at_turn: number;
  evicted_at_ts: number;
  last_accessed_ts: number | null;
  access_count: number;
}

export class PageHeapEntry {
  pageId: string;
  originalMessageId: string;
  originalType: string;
  rawContent: MessageContent;
  toolName: string | null;
  toolCallId: string | null;
  tokenCount: number;
  evictedAtTurn: number;
  evictedAtTs: number;
  lastAccessedTs: number | null;
  accessCount: number;

  constructor(fields: {
    pageId: string;
    originalMessageId: string;
    originalType: string;
    rawContent: MessageContent;
    toolName?: string | null;
    toolCallId?: string | null;
    tokenCount: number;
    evictedAtTurn: number;
    evictedAtTs?: number;
    lastAccessedTs?: number | null;
    accessCount?: number;
  }) {
    this.pageId = fields.pageId;
    this.originalMessageId = fields.originalMessageId;
    this.originalType = fields.originalType;
    this.rawContent = fields.rawContent;
    this.toolName = fields.toolName ?? null;
    this.toolCallId = fields.toolCallId ?? null;
    this.tokenCount = fields.tokenCount;
    this.evictedAtTurn = fields.evictedAtTurn;
    this.evictedAtTs = fields.evictedAtTs ?? Date.now() / 1_000;
    this.lastAccessedTs = fields.lastAccessedTs ?? null;
    this.accessCount = fields.accessCount ?? 0;
  }

  get lruTs(): number {
    return this.lastAccessedTs ?? this.evictedAtTs;
  }

  touch(): void {
    this.lastAccessedTs = Date.now() / 1_000;
    this.accessCount += 1;
  }

  toWire(): PageHeapEntryWire {
    return {
      page_id: this.pageId,
      original_message_id: this.originalMessageId,
      original_type: this.originalType,
      raw_content: this.rawContent,
      tool_name: this.toolName,
      tool_call_id: this.toolCallId,
      token_count: this.tokenCount,
      evicted_at_turn: this.evictedAtTurn,
      evicted_at_ts: this.evictedAtTs,
      last_accessed_ts: this.lastAccessedTs,
      access_count: this.accessCount,
    };
  }

  static fromWire(w: PageHeapEntryWire): PageHeapEntry {
    return new PageHeapEntry({
      pageId: w.page_id,
      originalMessageId: w.original_message_id,
      originalType: w.original_type,
      rawContent: w.raw_content,
      toolName: w.tool_name,
      toolCallId: w.tool_call_id,
      tokenCount: w.token_count,
      evictedAtTurn: w.evicted_at_turn,
      evictedAtTs: w.evicted_at_ts,
      lastAccessedTs: w.last_accessed_ts,
      accessCount: w.access_count,
    });
  }
}

// ---------------------------------------------------------------------------
// MarkCandidate — output of the Mark phase
// ---------------------------------------------------------------------------

export interface MarkCandidate {
  message: BaseMessage;
  messageIndex: number;
  tokenCount: number;
  turnAge: number;
  importanceScore: number;
  evictionReason: string;
}

// ---------------------------------------------------------------------------
// EvictionRecord — audit log entry
// ---------------------------------------------------------------------------

export interface EvictionRecord {
  pageId: string;
  originalMessageId: string;
  originalType: string;
  tokenCount: number;
  evictedAtTurn: number;
  evictedAtTs: number;
  policyUsed: EvictionPolicy;
}

// ---------------------------------------------------------------------------
// Pointer helpers
// ---------------------------------------------------------------------------

export function computePageId(message: BaseMessage): string {
  const payload = JSON.stringify(
    {
      id: message.id,
      type: message._getType(),
      content: message.content,
    },
    (_, v) => (v === undefined ? null : v)
  );
  return createHash("sha256").update(payload).digest("hex").slice(0, 16);
}

export function renderPointer(pageId: string, summary: string): string {
  return `<PAGE_FAULT_ID: ${pageId} | SUMMARY: ${summary}>`;
}

export function parsePointer(content: unknown): { pageId: string; summary: string } | null {
  if (typeof content !== "string") return null;
  const m = POINTER_RE.exec(content.trim());
  return m ? { pageId: m[1], summary: m[2] } : null;
}

export function isPointerContent(content: unknown): boolean {
  return parsePointer(content) !== null;
}

/**
 * Build a replacement message carrying the same `id` as `original`.
 *
 * LangGraph's messagesStateReducer deduplicates by id, so returning this
 * message from a node produces an in-place update — not an append.
 */
export function makePagePointerMessage(
  original: BaseMessage,
  pageId: string,
  summary: string,
  evictedAtTurn: number
): BaseMessage {
  const pointerContent = renderPointer(pageId, summary);
  const extraKwargs = {
    l1_page_id: pageId,
    l1_original_type: original._getType(),
    l1_evicted_at_turn: evictedAtTurn,
  };
  const originalKwargs = (original.additional_kwargs as Record<string, unknown>) ?? {};

  if (original instanceof ToolMessage) {
    return new ToolMessage({
      id: original.id,
      content: pointerContent,
      tool_call_id: original.tool_call_id,
      name: original.name,
      additional_kwargs: { ...originalKwargs, ...extraKwargs },
    });
  }

  // AIMessage and everything else
  return new AIMessage({
    id: original.id,
    content: pointerContent,
    additional_kwargs: { ...originalKwargs, ...extraKwargs },
  });
}

// ---------------------------------------------------------------------------
// L1PagerState — LangGraph Annotation (opt-in mixin)
// ---------------------------------------------------------------------------

/**
 * Optional state annotation for graphs that want full L1Pager observability.
 *
 * Usage:
 *   const MyState = Annotation.Root({
 *     ...L1PagerAnnotation.spec,
 *     myField: Annotation<string>({ default: () => "" }),
 *   });
 */
export const L1PagerAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  l1TurnCount: Annotation<number>({
    reducer: (prev, next) => next,
    default: () => 0,
  }),
  /** page_id -> PageHeapEntryWire — survives graph checkpointing */
  l1HeapIndex: Annotation<Record<string, PageHeapEntryWire>>({
    reducer: (prev, next) => ({ ...prev, ...next }),
    default: () => ({}),
  }),
  l1EvictionLog: Annotation<EvictionRecord[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
  /** Guards recursive demand-paging from infinite loops. */
  l1PageFaultDepth: Annotation<number>({
    reducer: (prev, next) => next,
    default: () => 0,
  }),
});

export type L1PagerState = typeof L1PagerAnnotation.State;
