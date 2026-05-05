/**
 * L1-Pager: Mark Phase (TypeScript)
 *
 * Mirrors python/l1_pager/mark.py exactly. O(n) scan, no I/O, <1ms
 * on 200-message arrays. Sorted output: highest eviction priority first.
 */

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  MessageContent,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  EvictionConfig,
  EvictionPolicy,
  MarkCandidate,
  isPointerContent,
} from "./schema.js";

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

const WHITESPACE_RE = /\s+/g;

/**
 * Fast token estimate: 1 token ≈ 4 chars for English text.
 * Handles string | MessageContent (array of content blocks).
 */
export function estimateTokens(content: MessageContent | unknown): number {
  if (typeof content === "string") {
    const collapsed = content.replace(WHITESPACE_RE, " ");
    return Math.max(1, Math.floor(collapsed.length / 4));
  }

  if (Array.isArray(content)) {
    let total = 0;
    for (const block of content) {
      if (typeof block === "string") {
        total += estimateTokens(block);
      } else if (typeof block === "object" && block !== null) {
        const b = block as Record<string, unknown>;
        total += estimateTokens(b["text"] ?? b["content"] ?? "");
      }
    }
    return Math.max(1, total);
  }

  if (typeof content === "object" && content !== null) {
    return estimateTokens(String(content));
  }

  return 1;
}

// ---------------------------------------------------------------------------
// Turn-age calculation
// ---------------------------------------------------------------------------

/**
 * Returns a per-message array of turn ages (0 = current turn).
 *
 * A "turn" increments on each HumanMessage after the first.
 * Mirrors python compute_turn_ages() exactly.
 */
export function computeTurnAges(messages: BaseMessage[]): number[] {
  const turnIndex: number[] = [];
  let current = 0;

  for (let i = 0; i < messages.length; i++) {
    if (i > 0 && messages[i] instanceof HumanMessage) {
      current += 1;
    }
    turnIndex.push(current);
  }

  const maxTurn = current;
  return turnIndex.map((t) => maxTurn - t);
}

// ---------------------------------------------------------------------------
// Ephemeral classification
// ---------------------------------------------------------------------------

const ERROR_KEYWORDS = ["error", "exception", "traceback", "failed", "panic", "fatal"] as const;
const JSON_START_CHARS = new Set(["{", "["]);

/**
 * True if the message is structurally ephemeral (eviction candidate).
 *
 * Non-evictable:
 *   - HumanMessage (user intent — must never be lost)
 *   - SystemMessage (graph invariants)
 *   - Pure tool-dispatch AIMessage (no content, only tool_calls)
 *   - Messages already holding a pointer (already swept)
 */
export function isEphemeral(message: BaseMessage): boolean {
  if (message instanceof HumanMessage || message instanceof SystemMessage) {
    return false;
  }

  if (isPointerContent(message.content)) {
    return false;
  }

  if (message instanceof ToolMessage) {
    return true;
  }

  if (message instanceof AIMessage) {
    const hasContent =
      typeof message.content === "string"
        ? message.content.length > 0
        : (message.content as unknown[]).length > 0;

    // Pure dispatcher: structural, keep it
    const isPureDispatcher = !hasContent && (message.tool_calls?.length ?? 0) > 0;
    return !isPureDispatcher && hasContent;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Importance scoring
// ---------------------------------------------------------------------------

/**
 * Heuristic importance in [0.0, 1.0]. Lower = evict sooner.
 * Mirrors python importance_score() weights exactly.
 */
export function importanceScore(message: BaseMessage, turnAge: number): number {
  let score = 0.7;

  const contentStr =
    typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content);

  const lowered = contentStr.toLowerCase();

  if (ERROR_KEYWORDS.some((kw) => lowered.includes(kw))) {
    score += 0.3;
  }

  const stripped = contentStr.trimStart();
  if (stripped.length > 0 && JSON_START_CHARS.has(stripped[0])) {
    score -= 0.2;
  }

  if (/\d{5,}/.test(contentStr)) {
    score -= 0.1;
  }

  score -= Math.min(0.5, turnAge * 0.05);

  return Math.max(0.0, Math.min(1.0, score));
}

// ---------------------------------------------------------------------------
// MarkPhase
// ---------------------------------------------------------------------------

export class MarkPhase {
  constructor(private readonly config: EvictionConfig) {}

  /**
   * O(n) scan over messages. Returns candidates sorted by eviction priority.
   *
   * Both criteria must be satisfied:
   *   tokenCount > minTokens   AND   turnAge > minTurnsOld
   */
  scan(messages: BaseMessage[], _currentTurn = 0): MarkCandidate[] {
    const ages = computeTurnAges(messages);
    const candidates: MarkCandidate[] = [];

    for (let idx = 0; idx < messages.length; idx++) {
      const msg = messages[idx];

      if (!isEphemeral(msg)) continue;

      const tokens = estimateTokens(msg.content);
      if (tokens <= this.config.minTokens) continue;

      const age = ages[idx];
      if (age <= this.config.minTurnsOld) continue;

      const imp = importanceScore(msg, age);
      candidates.push({
        message: msg,
        messageIndex: idx,
        tokenCount: tokens,
        turnAge: age,
        importanceScore: imp,
        evictionReason: this.reason(msg, tokens, age),
      });
    }

    this.sortCandidates(candidates);
    return candidates;
  }

  // ---- private helpers ----------------------------------------------------

  private sortCandidates(candidates: MarkCandidate[]): void {
    const policy: EvictionPolicy = this.config.policy;
    const w = this.config.hybridDecayWeight;

    if (policy === "lru") {
      candidates.sort((a, b) => b.turnAge - a.turnAge);
    } else if (policy === "importance") {
      candidates.sort((a, b) => a.importanceScore - b.importanceScore);
    } else {
      // hybrid
      candidates.sort(
        (a, b) =>
          a.importanceScore - a.turnAge * w - (b.importanceScore - b.turnAge * w)
      );
    }
  }

  private reason(msg: BaseMessage, tokens: number, age: number): string {
    return `${msg._getType()} | ~${tokens} tokens | ${age} turns old`;
  }
}
