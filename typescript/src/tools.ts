/**
 * L1-Pager: System tool definitions (TypeScript)
 * Mirrors python/l1_pager/tools.py.
 */

export const FETCH_EVICTED_PAGE_NAME = "fetch_evicted_page" as const;

/** JSON Schema for binding fetch_evicted_page to the model. */
export const FETCH_EVICTED_PAGE_SCHEMA = {
  name: FETCH_EVICTED_PAGE_NAME,
  description:
    "Retrieve the original content of a context page that was compressed and evicted " +
    "from the active context window to save space. " +
    "Use this when you see a <PAGE_FAULT_ID: {id} | SUMMARY: {summary}> reference " +
    "and need the full original content to answer accurately.",
  parameters: {
    type: "object" as const,
    properties: {
      page_id: {
        type: "string",
        description:
          "The hexadecimal page ID from the PAGE_FAULT_ID reference. " +
          "Example: if you see '<PAGE_FAULT_ID: abc123 | SUMMARY: ...>', pass 'abc123'.",
      },
    },
    required: ["page_id"],
  },
};

export function isPageFaultToolCall(toolCall: {
  name?: string;
  [key: string]: unknown;
}): boolean {
  return toolCall.name === FETCH_EVICTED_PAGE_NAME;
}
