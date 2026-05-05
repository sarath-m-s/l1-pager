/**
 * L1-Pager: Fast extractive summarizer (TypeScript)
 * Mirrors python/l1_pager/summarizer.py exactly.
 * Invariant: returned string is single-line, ≤140 chars.
 */

import { BaseMessage, ToolMessage } from "@langchain/core/messages";

const MAX_SUMMARY_CHARS = 140;
const SENTENCE_RE = /[.!?]\s/;

function firstSentence(text: string): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  const m = SENTENCE_RE.exec(collapsed);
  return m ? collapsed.slice(0, m.index + m[0].length).trim() : collapsed.slice(0, MAX_SUMMARY_CHARS);
}

function summarizeJson(text: string): string {
  try {
    const obj = JSON.parse(text.trim());
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
      const keys = Object.keys(obj).slice(0, 5).join(", ");
      return `JSON object with keys: ${keys}`;
    }
    if (Array.isArray(obj)) {
      return `JSON array with ${obj.length} items`;
    }
  } catch {
    // not valid JSON
  }
  return "";
}

export function generateSummary(message: BaseMessage): string {
  const rawContent = message.content;
  const contentStr = (typeof rawContent === "string" ? rawContent : JSON.stringify(rawContent)).trim();
  const charCount = contentStr.length;

  let raw: string;

  if (contentStr && (contentStr[0] === "{" || contentStr[0] === "[")) {
    const jsonSummary = summarizeJson(contentStr);
    raw = jsonSummary ? jsonSummary.slice(0, MAX_SUMMARY_CHARS) : `${message._getType()} (${charCount} chars)`;
  } else if (message instanceof ToolMessage && message.name) {
    const first = firstSentence(contentStr);
    raw = first.length > 15
      ? `${message.name}: ${first}`.slice(0, MAX_SUMMARY_CHARS)
      : `${message.name} output (${charCount} chars)`;
  } else {
    const first = firstSentence(contentStr);
    raw = first.length > 15 ? first.slice(0, MAX_SUMMARY_CHARS) : `${message._getType()} (${charCount} chars)`;
  }

  // Invariant: no newlines
  return raw.replace(/\s+/g, " ").trim();
}
