// State-machine runner. Walks a State to the `final` variant by dispatching
// on `state.kind`. The exhaustive switch + assertNever guarantees every
// variant is handled — a new state added to the union will fail to compile
// here until it has a transition.

import type { Final, State, Validated } from "./state.js";
import { assertNever, escalate } from "./state.js";
import { classify, type ClassifyContext } from "./nodes/classify.js";
import { draft } from "./nodes/draft.js";
import { retrieve, type RetrievalContext } from "./nodes/retrieve.js";
import { triage } from "./nodes/triage.js";
import { validate } from "./nodes/validate.js";
import { stepStart, stepDetail } from "./logger.js";

const MAX_DRAFT_RETRIES = 2;

export interface PipelineContext {
  retrieval: RetrievalContext;
  classify: ClassifyContext;
}

export async function runPipeline(
  initial: State,
  ctx: PipelineContext,
): Promise<Final> {
  let s: State = initial;
  // Cap iterations to keep retries from looping. With 6 stages and at most
  // MAX_DRAFT_RETRIES re-drafts, this is plenty.
  const MAX_STEPS = 16;
  const pipelineFinish = stepStart("pipeline", `ticket starting`);
  for (let step = 0; step < MAX_STEPS; step++) {
    const finish = stepStart("pipeline", `step ${step}: ${s.kind}`);
    switch (s.kind) {
      case "raw":
        s = await triage(s);
        break;
      case "triaged":
        s = await classify(s, ctx.classify);
        break;
      case "classified":
        s = await retrieve(s, ctx.retrieval);
        break;
      case "retrieved":
        s = await draft(s);
        break;
      case "drafted":
        s = await validate(s);
        break;
      case "validated":
        s = await transitionFromValidated(s);
        break;
      case "final":
        finish();
        stepDetail("pipeline", "result", `status=${s.status} type=${s.requestType} area=${s.productArea}`);
        pipelineFinish();
        return s;
      default:
        return assertNever(s);
    }
    finish();
  }
  pipelineFinish();
  throw new Error(`Pipeline did not terminate within ${MAX_STEPS} steps`);
}

async function transitionFromValidated(v: Validated): Promise<State> {
  const { validation, draft: d } = v;

  if (validation.kind === "ok") {
    return {
      kind: "final",
      input: v.raw.input,
      response: d.response,
      productArea: v.classification.productArea,
      status: v.classification.status,
      requestType: v.classification.requestType,
      justification: d.justification,
    };
  }

  if (validation.kind === "off_topic") {
    return escalate(
      v.raw.input,
      `Validator flagged off-topic response: ${validation.reason}`,
    );
  }

  // unsupported_claims path
  if (
    validation.suggestedAction === "escalate" ||
    d.retryCount >= MAX_DRAFT_RETRIES
  ) {
    return escalate(
      v.raw.input,
      `Could not produce a fully grounded response after ${d.retryCount} retries.`,
    );
  }

  // Retry: produce a new Drafted with retryCount+1 and the unsupported claims
  // fed back as feedback. The runner will then re-validate it.
  return draft(v, validation.claims);
}
