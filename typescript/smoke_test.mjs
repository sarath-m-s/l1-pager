/**
 * Smoke test — exercises every module end-to-end without a real LLM.
 * Run: node smoke_test.mjs
 */
import { createHash } from "crypto";

// ── imports from compiled dist ─────────────────────────────────────────────
import {
  DEFAULT_EVICTION_CONFIG,
  PageHeapEntry,
  computePageId,
  renderPointer,
  parsePointer,
  isPointerContent,
  makePagePointerMessage,
} from "./dist/schema.js";

import { estimateTokens, computeTurnAges, isEphemeral, MarkPhase } from "./dist/mark.js";
import { InMemoryHeap, buildHeap } from "./dist/heap.js";
import { generateSummary } from "./dist/summarizer.js";
import { SweepPhase } from "./dist/sweep.js";
import { FETCH_EVICTED_PAGE_NAME, isPageFaultToolCall } from "./dist/tools.js";
import { L1Pager } from "./dist/interceptor.js";
import { createL1PagerNode, createL1PagerReactAgent, L1PagerGraphAnnotation } from "./dist/langgraph_node.js";

import {
  AIMessage, HumanMessage, SystemMessage, ToolMessage,
} from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

const PASS = "\x1b[92m✓\x1b[0m";
const FAIL = "\x1b[91m✗\x1b[0m";
let failures = 0;

function check(label, value, expected) {
  const ok = expected === undefined ? !!value : value === expected;
  console.log(`  ${ok ? PASS : FAIL} ${label}`);
  if (!ok) { failures++; console.log(`      got: ${JSON.stringify(value)}, want: ${JSON.stringify(expected)}`); }
}

// ── 1. Schema ──────────────────────────────────────────────────────────────
console.log("\n\x1b[1mSchema\x1b[0m");
{
  const msg = new ToolMessage({ content: '{"txn":"TXN-001"}', tool_call_id: "x", id: "m1" });
  const pid = computePageId(msg);
  check("computePageId is 16 hex chars", /^[a-f0-9]{16}$/.test(pid));

  const rendered = renderPointer(pid, "Transaction data");
  check("renderPointer starts with PAGE_FAULT_ID", rendered.startsWith("<PAGE_FAULT_ID:"));

  const parsed = parsePointer(rendered);
  check("parsePointer returns object", parsed !== null);
  check("parsePointer.pageId matches", parsed?.pageId, pid);
  check("parsePointer.summary matches", parsed?.summary, "Transaction data");

  check("isPointerContent true for pointer", isPointerContent(rendered));
  check("isPointerContent false for plain text", isPointerContent("hello"), false);

  const ptr = makePagePointerMessage(msg, pid, "Transaction data", 5);
  check("pointer message keeps original id", ptr.id, "m1");
  check("pointer message content is pointer", isPointerContent(ptr.content));
  check("pointer is still ToolMessage", ptr instanceof ToolMessage);
}

// ── 2. Mark phase ──────────────────────────────────────────────────────────
console.log("\n\x1b[1mMark Phase\x1b[0m");
{
  const large = "x".repeat(2500);
  check("estimateTokens string", estimateTokens(large), 625);
  check("estimateTokens array blocks", estimateTokens([{ text: "a".repeat(400) }]), 100);

  const msgs = [
    new HumanMessage({ content: "h1", id: "1" }),
    new AIMessage({ content: "a1", id: "2" }),
    new HumanMessage({ content: "h2", id: "3" }),
    new AIMessage({ content: "a2", id: "4" }),
  ];
  const ages = computeTurnAges(msgs);
  check("turn ages h1=1,a1=1,h2=0,a2=0", JSON.stringify(ages), "[1,1,0,0]");

  const tool = new ToolMessage({ content: "data", tool_call_id: "t", id: "5" });
  check("ToolMessage isEphemeral", isEphemeral(tool));
  check("HumanMessage not ephemeral", !isEphemeral(msgs[0]));
  const dispatcher = new AIMessage({ content: "", tool_calls: [{ name: "fn", args: {}, id: "tc1" }], id: "6" });
  check("pure dispatcher AIMessage not ephemeral", !isEphemeral(dispatcher));

  // Build a conversation where turn-0 tool output ages past threshold
  const conv = [new SystemMessage({ content: "sys", id: "s0" })];
  const toolCallId = "tc-test";
  conv.push(new HumanMessage({ content: "Q0", id: "q0" }));
  conv.push(new AIMessage({ content: "", tool_calls: [{ name: "fetch", args: {}, id: toolCallId }], id: "a0" }));
  conv.push(new ToolMessage({ content: "x".repeat(3000), tool_call_id: toolCallId, name: "fetch", id: "t0" }));
  for (let i = 1; i < 6; i++) {
    conv.push(new HumanMessage({ content: `Q${i}`, id: `q${i}` }));
    conv.push(new AIMessage({ content: `A${i}`, id: `a${i}` }));
  }

  const cfg = { ...DEFAULT_EVICTION_CONFIG, minTokens: 100, minTurnsOld: 3 };
  const phase = new MarkPhase(cfg);
  const candidates = phase.scan(conv);
  check("MarkPhase finds the large ToolMessage", candidates.length >= 1);
  check("candidate token count > 100", candidates[0]?.tokenCount > 100);
  check("candidate turn age > 3", candidates[0]?.turnAge > 3);
}

// ── 3. Heap ─────────────────────────────────────────────────────────────────
console.log("\n\x1b[1mInMemoryHeap\x1b[0m");
{
  const cfg = { ...DEFAULT_EVICTION_CONFIG, maxHeapEntries: 3 };
  const heap = new InMemoryHeap(cfg);

  const entry = new PageHeapEntry({
    pageId: "abc123def45678",
    originalMessageId: "m1",
    originalType: "ToolMessage",
    rawContent: '{"needle":42}',
    tokenCount: 600,
    evictedAtTurn: 0,
  });

  await heap.put(entry);
  check("size after put", await heap.size(), 1);

  const got = await heap.get("abc123def45678");
  check("get returns entry", got !== null);
  check("raw content preserved", got?.rawContent, '{"needle":42}');
  check("access_count incremented", got?.accessCount, 1);

  await heap.delete("abc123def45678");
  check("get after delete returns null", await heap.get("abc123def45678"), null);

  // LRU overflow: put 4 entries into a heap with maxHeapEntries=3
  for (let i = 0; i < 4; i++) {
    await heap.put(new PageHeapEntry({
      pageId: `overflow${i.toString().padStart(8, "0")}`,
      originalMessageId: `m${i}`,
      originalType: "ToolMessage",
      rawContent: `data${i}`,
      tokenCount: 600,
      evictedAtTurn: i,
    }));
  }
  check("LRU cap respected (size=3)", await heap.size(), 3);
  check("oldest entry evicted", await heap.get("overflow00000000"), null);
  check("newest entry retained", (await heap.get("overflow00000003")) !== null);
}

// ── 4. Summarizer ──────────────────────────────────────────────────────────
console.log("\n\x1b[1mSummarizer\x1b[0m");
{
  const toolMsg = new ToolMessage({
    content: '{"transaction_id":"TXN-001","amount":9999}',
    tool_call_id: "x",
    name: "lookup_txn",
    id: "m1",
  });
  const summary = generateSummary(toolMsg);
  check("summary ≤140 chars", summary.length <= 140);
  check("summary is single line", !summary.includes("\n"));
  check("summary non-empty", summary.length > 0);

  const aiMsg = new AIMessage({ content: "Let me think through this step by step.", id: "m2" });
  const aiSummary = generateSummary(aiMsg);
  check("AI message summary single line", !aiSummary.includes("\n"));
}

// ── 5. Sweep ────────────────────────────────────────────────────────────────
console.log("\n\x1b[1mSweep Phase\x1b[0m");
{
  const NEEDLE = JSON.stringify({ txn_id: "TXN-SWEEP-001", amount: 12345.67 });
  const padding = Array.from({ length: 120 }, (_, i) => `LOG ${i}: event`).join("\n");
  const largeContent = NEEDLE + "\n\n" + padding;
  const toolMsgId = "tool-sweep-001";

  const msgs = [
    new HumanMessage({ content: "Pull report", id: "h0" }),
    new AIMessage({ content: "", tool_calls: [{ name: "pull", args: {}, id: "tc0" }], id: "a0" }),
    new ToolMessage({ content: largeContent, tool_call_id: "tc0", name: "pull", id: toolMsgId }),
    ...Array.from({ length: 5 }, (_, i) => [
      new HumanMessage({ content: `Q${i+1}`, id: `h${i+1}` }),
      new AIMessage({ content: `A${i+1}`, id: `a${i+1}` }),
    ]).flat(),
  ];

  const cfg = { ...DEFAULT_EVICTION_CONFIG, minTokens: 100, minTurnsOld: 3 };
  const heap = buildHeap(cfg);
  const sweep = new SweepPhase(cfg, heap);
  const result = await sweep.run(msgs, 10);

  check("sweep evicted at least 1 message", result.pagesWritten >= 1);
  check("tokens freed > 0", result.tokensFreed > 0);
  check("message count preserved", result.updatedMessages.length === msgs.length);

  const sweptMsg = result.updatedMessages.find(m => m.id === toolMsgId);
  check("swept message is pointer", isPointerContent(sweptMsg?.content));
  check("pointer message id unchanged (add_messages invariant)", sweptMsg?.id, toolMsgId);

  // Verify raw content survived in heap
  const parsed = parsePointer(sweptMsg.content);
  const heapEntry = await heap.get(parsed.pageId);
  check("heap entry exists", heapEntry !== null);
  check("raw content byte-identical", heapEntry?.rawContent, largeContent);

  // Second sweep must be idempotent (pointers are not ephemeral)
  const result2 = await sweep.run(result.updatedMessages, 10);
  check("second sweep is no-op", result2.pagesWritten, 0);
}

// ── 6. Interceptor (mock model) ────────────────────────────────────────────
console.log("\n\x1b[1mL1Pager Interceptor\x1b[0m");
{
  const NEEDLE_AMOUNT = 7628.86;
  const NEEDLE_TXN    = "TXN-MOCK-0042";

  // Build a pre-swept conversation
  const content = JSON.stringify({ txn: NEEDLE_TXN, amount: NEEDLE_AMOUNT }) + "\n" + "x".repeat(2400);
  const toolMsgId = "t-intercept-001";
  const msgs = [
    new HumanMessage({ content: "Pull data", id: "h0" }),
    new AIMessage({ content: "", tool_calls: [{ name: "pull", args: {}, id: "tc0" }], id: "a0" }),
    new ToolMessage({ content, tool_call_id: "tc0", name: "pull", id: toolMsgId }),
    ...Array.from({ length: 5 }, (_, i) => [
      new HumanMessage({ content: `Q${i+1}`, id: `h${i+1}` }),
      new AIMessage({ content: `A${i+1}`, id: `a${i+1}` }),
    ]).flat(),
  ];

  const cfg = { ...DEFAULT_EVICTION_CONFIG, minTokens: 100, minTurnsOld: 3 };
  const heap = buildHeap(cfg);

  // Pre-sweep to populate heap
  const sweep = new SweepPhase(cfg, heap);
  const preSwept = await sweep.run(msgs, 10);
  const ptrMsg = preSwept.updatedMessages.find(m => m.id === toolMsgId);
  const parsed = parsePointer(ptrMsg.content);
  const pageId = parsed.pageId;

  // Mock model: call 1 → page fault, call 2 → answer with exact value
  let callIdx = 0;
  const mockModel = {
    bindTools: function() { return this; },
    invoke: async function(messages) {
      callIdx++;
      if (callIdx === 1) {
        return new AIMessage({
          content: "",
          tool_calls: [{ name: FETCH_EVICTED_PAGE_NAME, args: { page_id: pageId }, id: "pf1" }],
          id: `ai-${callIdx}`,
        });
      }
      // Verify re-injected content is present in messages
      const injected = messages.find(m => m instanceof ToolMessage && m.name === FETCH_EVICTED_PAGE_NAME);
      const hasNeedle = injected && String(injected.content).includes(NEEDLE_TXN);
      return new AIMessage({
        content: hasNeedle
          ? `The amount was $${NEEDLE_AMOUNT.toFixed(2)} for ${NEEDLE_TXN}.`
          : "I could not find the data.",
        id: `ai-${callIdx}`,
      });
    },
  };

  const pager = new L1Pager(mockModel, heap, cfg);
  const question = new HumanMessage({ content: "Exact amount for TXN-MOCK-0042?", id: "q-final" });
  const result = await pager.ainvoke([...preSwept.updatedMessages, question], 10);

  check("page fault resolved", result.pageFaults, 1);
  check("model called twice", callIdx, 2);
  check("correct amount recalled", result.aiMessage.content.includes(NEEDLE_AMOUNT.toFixed(2)));
  check("correct TXN ID in answer", result.aiMessage.content.includes(NEEDLE_TXN));
  check("output messages include pointer updates", result.messages.length > 0);
}

// ── 7. LangGraph node (Command routing) ───────────────────────────────────
console.log("\n\x1b[1mLangGraph Node & Graph\x1b[0m");
{
  let callCount = 0;
  const mockModel = {
    bindTools: function() { return this; },
    invoke: async function() {
      callCount++;
      return new AIMessage({ content: "Done.", id: `ai-graph-${callCount}` });
    },
  };

  const cfg = DEFAULT_EVICTION_CONFIG;
  const heap = buildHeap(cfg);
  const pager = new L1Pager(mockModel, heap, cfg);
  const node = createL1PagerNode(pager);

  // Simulate node call
  const state = {
    messages: [new HumanMessage({ content: "Hello", id: "h1" })],
    l1TurnCount: 0,
    l1EvictionLog: [],
    l1PageFaultTotal: 0,
    l1TokensFreedTotal: 0,
  };

  const cmd = await node(state);
  check("node returns Command", cmd instanceof Command);
  check("turn count incremented", cmd.update.l1TurnCount, 1);
  check("messages in update", Array.isArray(cmd.update.messages));

  // Full graph round-trip
  const graph = createL1PagerReactAgent(mockModel, []);
  const graphResult = await graph.invoke({
    messages: [new HumanMessage({ content: "Test", id: "g1" })],
  });
  check("graph returns messages", Array.isArray(graphResult.messages));
  check("graph last message is AIMessage", graphResult.messages.at(-1) instanceof AIMessage);
}

// ── Summary ────────────────────────────────────────────────────────────────
console.log(`\n${"─".repeat(50)}`);
if (failures === 0) {
  console.log("\x1b[1m\x1b[92m  ✅  All TypeScript smoke tests passed\x1b[0m\n");
} else {
  console.log(`\x1b[1m\x1b[91m  ❌  ${failures} check(s) failed\x1b[0m\n`);
  process.exit(1);
}
