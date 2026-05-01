// Validate node — evidence-based check that every factual claim in the draft
// is supported by at least one retrieved chunk. Maps the LLM's verdict into
// the typed ValidationResult variants the runner consumes.

import { z } from "zod";

import { MODELS, structured } from "../llm.js";
import type { Drafted, Validated, ValidationResult } from "../state.js";
import type { ScoredChunk } from "../types.js";

const MAX_RETRIES = 2;

const ValidateLlmSchema = z.object({
  ok: z.boolean(),
  unsupported_claims: z.array(z.string()),
  off_topic_reason: z.string().nullable(),
});

const SYSTEM = `You are a fact-checker for a customer-support response. You will receive a draft response, the original ticket, and the source documentation chunks the response was supposed to be grounded in.

Your job:

1. Identify any factual claim in the response that is NOT directly supported by the source chunks. A "claim" includes specific procedures, policies, prices, contact numbers, URLs, time limits, or step-by-step instructions. Generic phrasings ("Hi", "I hope this helps", "Thank you for reaching out") are not claims.

2. Check whether the response is on-topic relative to the original ticket. If the response addresses something completely different from what the user asked, fill off_topic_reason with a one-sentence explanation; otherwise leave it null.

3. If everything checks out, set ok=true and return empty unsupported_claims.

A response that says "I don't have specific guidance on this" is acceptable when the chunks really don't contain the answer — that is honest, not unsupported.`;

function formatChunks(chunks: ScoredChunk[]): string {
  if (chunks.length === 0) return "(none)";
  const lines: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]!;
    const body = c.content.length > 3500 ? c.content.slice(0, 3500) + "…" : c.content;
    lines.push(`[${i + 1}] path: ${c.path}`);
    lines.push(`    title: ${c.title}`);
    lines.push(body);
    lines.push("");
  }
  return lines.join("\n");
}

function buildUserPrompt(d: Drafted): string {
  const lines: string[] = [];
  lines.push(`Original ticket subject: ${d.raw.input.subject || "(none)"}`);
  lines.push(`Original ticket body: ${d.raw.input.issue}`);
  lines.push("");
  lines.push(`Draft response: ${d.response}`);
  lines.push(`Draft justification: ${d.justification}`);
  lines.push("");
  lines.push("Source documentation chunks:");
  lines.push(formatChunks(d.chunks));
  return lines.join("\n");
}

export async function validate(d: Drafted): Promise<Validated> {
  let result: z.infer<typeof ValidateLlmSchema>;
  try {
    result = await structured({
      model: MODELS.validate(),
      system: SYSTEM,
      user: buildUserPrompt(d),
      schema: ValidateLlmSchema,
    });
  } catch (e) {
    // If the validator fails, treat the draft as ok rather than escalating —
    // the validator itself failing is not the user's fault.
    return wrapValidated(d, { kind: "ok" });
  }

  let validation: ValidationResult;
  if (result.off_topic_reason) {
    validation = { kind: "off_topic", reason: result.off_topic_reason };
  } else if (!result.ok || result.unsupported_claims.length > 0) {
    const suggestedAction =
      d.retryCount >= MAX_RETRIES ? "escalate" : "retry";
    validation = {
      kind: "unsupported_claims",
      claims: result.unsupported_claims,
      suggestedAction,
    };
  } else {
    validation = { kind: "ok" };
  }

  return wrapValidated(d, validation);
}

function wrapValidated(d: Drafted, validation: ValidationResult): Validated {
  return {
    kind: "validated",
    raw: d.raw,
    flags: d.flags,
    classification: d.classification,
    chunks: d.chunks,
    draft: {
      raw: d.raw,
      flags: d.flags,
      classification: d.classification,
      chunks: d.chunks,
      response: d.response,
      justification: d.justification,
      retryCount: d.retryCount,
    },
    validation,
  };
}
