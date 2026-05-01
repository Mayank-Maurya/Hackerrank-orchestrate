// Shared domain types and the closed-set values the rubric requires.

export const STATUS = ["replied", "escalated"] as const;
export type Status = (typeof STATUS)[number];

export const REQUEST_TYPE = [
  "product_issue",
  "feature_request",
  "bug",
  "invalid",
] as const;
export type RequestType = (typeof REQUEST_TYPE)[number];

export const COMPANY = ["HackerRank", "Claude", "Visa", "None"] as const;
export type Company = (typeof COMPANY)[number];

// Resolved company after triage/classify. "None" is no longer allowed once we
// commit to a routing decision, but we keep "Unknown" for the genuinely
// ambiguous escalation path.
export const RESOLVED_COMPANY = ["HackerRank", "Claude", "Visa", "Unknown"] as const;
export type ResolvedCompany = (typeof RESOLVED_COMPANY)[number];

// Input row from support_tickets.csv (or the sample). Header casing varies
// between files; csv.ts normalizes to these field names.
export interface InputRow {
  issue: string;
  subject: string;
  company: Company;
}

// Output row written to support_tickets/output.csv. Field order matches the
// existing header in that file.
export interface OutputRow {
  issue: string;
  subject: string;
  company: string;
  response: string;
  product_area: string;
  status: Status;
  request_type: RequestType;
  justification: string;
}

// A chunk retrieved from the corpus.
export interface Chunk {
  // Path relative to data/, e.g. "claude/privacy-and-legal/12345-foo.md".
  path: string;
  // The company this chunk belongs to, derived from the top-level folder.
  company: Exclude<ResolvedCompany, "Unknown">;
  // The product_area folder this chunk belongs to (e.g. "screen", "privacy").
  productArea: string;
  // Title (first H1 or filename stem).
  title: string;
  // Chunk content.
  content: string;
}

// A retrieved chunk with its fused score.
export interface ScoredChunk extends Chunk {
  score: number;
}
