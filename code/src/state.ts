// The state machine. Every state is a variant of a discriminated union keyed
// on `kind`. Transitions are functions whose return type is also a State; the
// runner in pipeline.ts walks states until it reaches `final`.
//
// TypeScript's exhaustiveness checking on `switch (state.kind)` guarantees no
// state is silently skipped or forgotten.

import type {
  Chunk,
  Company,
  InputRow,
  RequestType,
  ResolvedCompany,
  ScoredChunk,
  Status,
} from "./types.js";

// ---------- Stage 0: raw input ------------------------------------------------
export interface RawTicket {
  kind: "raw";
  input: InputRow;
}

// ---------- Stage 1: triage (safety + cheap intent gate) ----------------------
export interface TriageFlags {
  hasInjection: boolean;        // prompt-injection patterns
  isSensitive: boolean;         // fraud/identity-theft/security-vuln/refund/score-dispute
  isGreeting: boolean;          // "thanks", "hello"
  isOutOfScope: boolean;        // unrelated to any of the three companies
  notes: string;                // why we set the flags above
}

export interface Triaged {
  kind: "triaged";
  raw: RawTicket;
  flags: TriageFlags;
}

// ---------- Stage 2: classify (route + provisional status) --------------------
export interface Classification {
  company: ResolvedCompany;
  productArea: string;          // closed list per company; "" for escalated
  requestType: RequestType;
  status: Status;               // provisional; can flip to escalated downstream
  reasoning: string;
}

export interface Classified {
  kind: "classified";
  raw: RawTicket;
  flags: TriageFlags;
  classification: Classification;
}

// ---------- Stage 3: retrieve -------------------------------------------------
export interface Retrieved {
  kind: "retrieved";
  raw: RawTicket;
  flags: TriageFlags;
  classification: Classification;
  chunks: ScoredChunk[];
}

// ---------- Stage 4: draft ----------------------------------------------------
export interface Drafted {
  kind: "drafted";
  raw: RawTicket;
  flags: TriageFlags;
  classification: Classification;
  chunks: ScoredChunk[];
  response: string;
  justification: string;
  retryCount: number;            // increments on each re-draft
}

// ---------- Stage 5: validate -------------------------------------------------
export type ValidationResult =
  | { kind: "ok" }
  | {
      kind: "unsupported_claims";
      claims: string[];
      suggestedAction: "retry" | "escalate";
    }
  | { kind: "off_topic"; reason: string };

export interface Validated {
  kind: "validated";
  raw: RawTicket;
  flags: TriageFlags;
  classification: Classification;
  chunks: ScoredChunk[];
  draft: Omit<Drafted, "kind">;
  validation: ValidationResult;
}

// ---------- Terminal stage ----------------------------------------------------
export interface Final {
  kind: "final";
  // Verbatim from input so we preserve the exact issue/subject/company strings.
  input: InputRow;
  // Output payload.
  response: string;
  productArea: string;
  status: Status;
  requestType: RequestType;
  justification: string;
}

// ---------- Union --------------------------------------------------------------
export type State =
  | RawTicket
  | Triaged
  | Classified
  | Retrieved
  | Drafted
  | Validated
  | Final;

// ---------- Helpers ------------------------------------------------------------
export function assertNever(x: never): never {
  throw new Error(`Unhandled state variant: ${JSON.stringify(x)}`);
}

// Convenience constructor for the common "early-exit to escalation" path.
export function escalate(
  input: InputRow,
  reason: string,
  requestType: RequestType = "product_issue",
): Final {
  return {
    kind: "final",
    input,
    response: "Escalate to a human",
    productArea: "",
    status: "escalated",
    requestType,
    justification: reason,
  };
}

// Convenience constructor for "out-of-scope but reply politely" path.
export function replyInvalid(
  input: InputRow,
  response: string,
  productArea: string,
  justification: string,
): Final {
  return {
    kind: "final",
    input,
    response,
    productArea,
    status: "replied",
    requestType: "invalid",
    justification,
  };
}
