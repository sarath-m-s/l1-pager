/**
 * L1-Pager: LangGraph integration — node factory and graph builder (TypeScript)
 *
 * LangGraph JS 0.2.x uses Command objects for conditional routing instead of
 * addConditionalEdges. Nodes declare their possible destinations via the
 * `ends` option in addNode(), then return Command({goto: destination}).
 */

import { AIMessage, BaseMessage } from "@langchain/core/messages";
import {
  Annotation,
  Command,
  END,
  messagesStateReducer,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AbstractHeap, InMemoryHeap, buildHeap } from "./heap.js";
import { L1Pager, L1PagerResponse } from "./interceptor.js";
import {
  DEFAULT_EVICTION_CONFIG,
  EvictionConfig,
  EvictionRecord,
} from "./schema.js";
import { FETCH_EVICTED_PAGE_NAME, FETCH_EVICTED_PAGE_SCHEMA } from "./tools.js";

// ---------------------------------------------------------------------------
// Graph state annotation
// ---------------------------------------------------------------------------

export const L1PagerGraphAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  l1TurnCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
  l1EvictionLog: Annotation<EvictionRecord[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
  l1PageFaultTotal: Annotation<number>({
    reducer: (prev, next) => prev + next,
    default: () => 0,
  }),
  l1TokensFreedTotal: Annotation<number>({
    reducer: (prev, next) => prev + next,
    default: () => 0,
  }),
});

export type L1PagerGraphState = typeof L1PagerGraphAnnotation.State;

// ---------------------------------------------------------------------------
// Routing helpers
// ---------------------------------------------------------------------------

function hasRealToolCalls(message: BaseMessage): boolean {
  const toolCalls: any[] = (message as any).tool_calls ?? [];
  return toolCalls.some((tc) => tc.name !== FETCH_EVICTED_PAGE_NAME);
}

// ---------------------------------------------------------------------------
// Node factory
// ---------------------------------------------------------------------------

/**
 * Returns a LangGraph node function that runs the full L1Pager pipeline and
 * routes to "tools" or END using Command objects (LangGraph JS 0.2 pattern).
 *
 * Register it with:
 *   builder.addNode("agent", createL1PagerNode(pager), { ends: ["tools", END] })
 */
export function createL1PagerNode(pager: L1Pager) {
  return async (state: L1PagerGraphState): Promise<Command> => {
    const messages = state.messages ?? [];
    const turn = state.l1TurnCount ?? 0;

    const result: L1PagerResponse = await pager.ainvoke(messages, turn);

    const updates: Partial<L1PagerGraphState> = {
      messages: result.messages,
      l1TurnCount: turn + 1,
      l1PageFaultTotal: result.pageFaults,
      l1TokensFreedTotal: result.tokensFreed,
      ...(result.evictionRecords.length > 0
        ? { l1EvictionLog: result.evictionRecords }
        : {}),
    };

    // Determine next node: real tool calls → "tools", otherwise end
    const lastMsg = result.messages[result.messages.length - 1];
    const goto =
      lastMsg instanceof AIMessage && hasRealToolCalls(lastMsg) ? "tools" : END;

    return new Command({ goto, update: updates });
  };
}

// ---------------------------------------------------------------------------
// Graph factory
// ---------------------------------------------------------------------------

export interface L1PagerReactAgentOptions {
  heap?: AbstractHeap;
  config?: EvictionConfig;
}

export function createL1PagerReactAgent(
  model: any,
  tools: any[],
  options: L1PagerReactAgentOptions = {}
) {
  const cfg = options.config ?? DEFAULT_EVICTION_CONFIG;
  const heap = options.heap ?? buildHeap(cfg);

  // Bind user tools + fetch_evicted_page to the model
  const userToolNames = new Set(tools.map((t: any) => t.name));
  const allTools = [...tools];
  if (!userToolNames.has(FETCH_EVICTED_PAGE_NAME)) {
    allTools.push(FETCH_EVICTED_PAGE_SCHEMA);
  }

  const pager = new L1Pager(model, heap, cfg);
  if (typeof model.bindTools === "function") {
    (pager as any).boundModel = model.bindTools(allTools);
  }

  const agentNode = createL1PagerNode(pager);

  // ToolNode routes back to agent after executing user tools
  const userToolsOnly = tools.filter(
    (t: any) => t.name !== FETCH_EVICTED_PAGE_NAME
  );
  const toolNode = new ToolNode(userToolsOnly);

  return new StateGraph(L1PagerGraphAnnotation)
    .addNode("agent", agentNode, { ends: ["tools", END] })
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addEdge("tools", "agent")
    .compile();
}
