/**
 * L1-Pager: The Interceptor (TypeScript)
 * Mirrors python/l1_pager/interceptor.py.
 *
 * Usage:
 *   const model = new ChatOpenAI({ model: "gpt-4o" });
 *   const heap = buildHeap(DEFAULT_EVICTION_CONFIG);
 *   const pager = new L1Pager({ model, heap });
 *
 *   // In your LangGraph node:
 *   const result = await pager.ainvoke(state.messages);
 *   return { messages: result.messages, l1EvictionLog: result.evictionRecords };
 */

import { AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { AbstractHeap, buildHeap } from "./heap.js";
import {
  DEFAULT_EVICTION_CONFIG,
  EvictionConfig,
  EvictionRecord,
  PageHeapEntry,
} from "./schema.js";
import { SweepPhase } from "./sweep.js";
import {
  FETCH_EVICTED_PAGE_NAME,
  FETCH_EVICTED_PAGE_SCHEMA,
  isPageFaultToolCall,
} from "./tools.js";

// ---------------------------------------------------------------------------
// Response type
// ---------------------------------------------------------------------------

export interface L1PagerResponse {
  aiMessage: AIMessage;
  /** Pointer replacements + final AI message — pass to messagesStateReducer. */
  messages: BaseMessage[];
  evictionRecords: EvictionRecord[];
  pageFaults: number;
  tokensFreed: number;
}

// ---------------------------------------------------------------------------
// Model interface (duck-typed for flexibility)
// ---------------------------------------------------------------------------

export interface ChatModelLike {
  invoke(messages: BaseMessage[], options?: unknown): Promise<AIMessage> | AIMessage;
  bindTools?(tools: unknown[], options?: unknown): ChatModelLike;
}

// ---------------------------------------------------------------------------
// L1Pager
// ---------------------------------------------------------------------------

export class L1Pager {
  private boundModel: ChatModelLike;
  private sweep: SweepPhase;

  constructor(
    private readonly model: ChatModelLike,
    private readonly heap: AbstractHeap = buildHeap(DEFAULT_EVICTION_CONFIG),
    private readonly config: EvictionConfig = DEFAULT_EVICTION_CONFIG
  ) {
    this.sweep = new SweepPhase(config, heap);
    // Bind the page-fault tool to the model if the model supports it
    this.boundModel =
      typeof model.bindTools === "function"
        ? model.bindTools([FETCH_EVICTED_PAGE_SCHEMA])
        : model;
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  async ainvoke(
    messages: BaseMessage[],
    currentTurn = 0,
    options?: unknown
  ): Promise<L1PagerResponse> {
    // 1. Sweep
    const sweepResult = await this.sweep.run(messages, currentTurn);
    const swept = sweepResult.updatedMessages;

    // 2. Initial model call
    let aiResponse = await this.boundModel.invoke(swept, options);
    let currentMessages = swept;
    let totalFaults = 0;

    // 3. Page-fault recovery loop
    while (this.hasPageFault(aiResponse)) {
      if (totalFaults >= this.config.maxPageFaultDepth) {
        console.warn(
          `[L1Pager] max_page_fault_depth (${this.config.maxPageFaultDepth}) reached, stopping recovery`
        );
        break;
      }
      [aiResponse, currentMessages] = await this.recoverPageFault(
        currentMessages,
        aiResponse,
        options
      );
      totalFaults++;
    }

    return {
      aiMessage: aiResponse,
      messages: [...sweepResult.updatedMessages, aiResponse],
      evictionRecords: sweepResult.evictionRecords,
      pageFaults: totalFaults,
      tokensFreed: sweepResult.tokensFreed,
    };
  }

  // ------------------------------------------------------------------
  // Page-fault recovery
  // ------------------------------------------------------------------

  private hasPageFault(response: AIMessage): boolean {
    const toolCalls: unknown[] = (response as any).tool_calls ?? [];
    return toolCalls.some(
      (tc) => typeof tc === "object" && tc !== null && isPageFaultToolCall(tc as any)
    );
  }

  private async recoverPageFault(
    messages: BaseMessage[],
    aiResponse: AIMessage,
    options?: unknown
  ): Promise<[AIMessage, BaseMessage[]]> {
    const toolCalls: any[] = (aiResponse as any).tool_calls ?? [];
    const faultCalls = toolCalls.filter((tc) => isPageFaultToolCall(tc));

    const fetchPage = async (tc: any): Promise<ToolMessage> => {
      const pageId: string = tc.args?.page_id ?? "";
      const entry: PageHeapEntry | null = await this.heap.get(pageId);

      let content: string;
      if (!entry) {
        content =
          `[L1Pager: page '${pageId}' not found — ` +
          `it may have expired (TTL=${this.config.entryTtlSeconds}s) or the page_id is incorrect.]`;
      } else {
        content = typeof entry.rawContent === "string"
          ? entry.rawContent
          : JSON.stringify(entry.rawContent);
      }

      return new ToolMessage({
        content,
        tool_call_id: tc.id,
        name: FETCH_EVICTED_PAGE_NAME,
      });
    };

    const toolMessages = await Promise.all(faultCalls.map(fetchPage));
    const extended = [...messages, aiResponse, ...toolMessages];
    const newResponse = await this.boundModel.invoke(extended, options);
    return [newResponse, extended];
  }
}
