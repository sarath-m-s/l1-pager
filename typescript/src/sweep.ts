/**
 * L1-Pager: Sweep Phase (TypeScript)
 * Mirrors python/l1_pager/sweep.py exactly.
 */

import { BaseMessage } from "@langchain/core/messages";
import { AbstractHeap } from "./heap.js";
import { MarkPhase } from "./mark.js";
import {
  EvictionConfig,
  EvictionRecord,
  MarkCandidate,
  PageHeapEntry,
  computePageId,
  makePagePointerMessage,
} from "./schema.js";
import { generateSummary } from "./summarizer.js";

export interface SweepResult {
  /** Full message list with pointer replacements. Pass to messagesStateReducer. */
  updatedMessages: BaseMessage[];
  evictionRecords: EvictionRecord[];
  tokensFreed: number;
  pagesWritten: number;
}

export class SweepPhase {
  private mark: MarkPhase;

  constructor(
    private readonly config: EvictionConfig,
    private readonly heap: AbstractHeap
  ) {
    this.mark = new MarkPhase(config);
  }

  async run(messages: BaseMessage[], currentTurn = 0): Promise<SweepResult> {
    const candidates = this.mark.scan(messages, currentTurn);

    if (candidates.length === 0) {
      return {
        updatedMessages: [...messages],
        evictionRecords: [],
        tokensFreed: 0,
        pagesWritten: 0,
      };
    }

    const entries = candidates.map((c) => this.buildEntry(c, currentTurn));

    // Concurrent heap writes
    await Promise.all(entries.map((e) => this.heap.put(e)));

    // Build pointer-replaced message list
    const evictIndex = new Map<number, [MarkCandidate, PageHeapEntry]>(
      candidates.map((c, i) => [c.messageIndex, [c, entries[i]]])
    );

    const updated: BaseMessage[] = messages.map((msg, idx) => {
      const pair = evictIndex.get(idx);
      if (!pair) return msg;
      const [_candidate, entry] = pair;
      const summary = generateSummary(msg);
      return makePagePointerMessage(msg, entry.pageId, summary, currentTurn);
    });

    const records: EvictionRecord[] = entries.map((e) => ({
      pageId: e.pageId,
      originalMessageId: e.originalMessageId,
      originalType: e.originalType,
      tokenCount: e.tokenCount,
      evictedAtTurn: currentTurn,
      evictedAtTs: Date.now() / 1_000,
      policyUsed: this.config.policy,
    }));

    return {
      updatedMessages: updated,
      evictionRecords: records,
      tokensFreed: candidates.reduce((s, c) => s + c.tokenCount, 0),
      pagesWritten: entries.length,
    };
  }

  private buildEntry(candidate: MarkCandidate, currentTurn: number): PageHeapEntry {
    const msg = candidate.message;
    return new PageHeapEntry({
      pageId: computePageId(msg),
      originalMessageId: msg.id ?? "",
      originalType: msg._getType(),
      rawContent: msg.content,
      toolName: (msg as any).name ?? null,
      toolCallId: (msg as any).tool_call_id ?? null,
      tokenCount: candidate.tokenCount,
      evictedAtTurn: currentTurn,
    });
  }
}
