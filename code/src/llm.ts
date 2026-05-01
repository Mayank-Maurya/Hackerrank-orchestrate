// OpenRouter client + structured-output helper.
//
// We use the `openai` SDK pointing to OpenRouter with `response_format: { type: "json_object" }`
// and a JSON Schema (converted from a Zod schema) injected into the system prompt for structured output. 
// The SDK returns raw JSON text; we then parse with Zod for runtime validation.

import OpenAI from "openai";
import type { z, ZodTypeAny } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

import { stepStart, logSent, logReceived, stepError } from "./logger.js";

let _client: OpenAI | null = null;

export function client(): OpenAI {
  if (_client) return _client;

  const apiKey = process.env.OPEN_ROUTER_KEY;
  if (!apiKey) {
    throw new Error(
      "OPEN_ROUTER_KEY not set. Copy code/.env.example to code/.env and fill it in.",
    );
  }

  _client = new OpenAI({
    apiKey,
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "HackerRank Orchestrate",
    }
  });

  return _client;
}

export const MODELS = {
  triage: () => process.env.OPENROUTER_TRIAGE_MODEL ?? "google/gemini-2.5-flash",
  classify: () => process.env.OPENROUTER_CLASSIFY_MODEL ?? "google/gemini-2.5-flash",
  draft: () => process.env.OPENROUTER_DRAFT_MODEL ?? "google/gemini-2.5-flash",
  validate: () => process.env.OPENROUTER_VALIDATE_MODEL ?? "google/gemini-2.5-flash",
};

// zod-to-json-schema can emit `$ref`/`definitions` blocks. We inline everything
// and strip the `$schema` field.
function toJsonSchema(schema: ZodTypeAny): Record<string, unknown> {
  const j = zodToJsonSchema(schema, { $refStrategy: "none" }) as Record<
    string,
    unknown
  >;
  delete j.$schema;
  return j;
}

// We serialize all calls through a single gate with a small delay to avoid rate limit issues.
const RATE_LIMIT_MS = 2000;
let lastCallEndMs = 0;
let chainTail: Promise<void> = Promise.resolve();

async function rateLimit(): Promise<void> {
  const myTurn = chainTail.then(async () => {
    const wait = Math.max(0, lastCallEndMs + RATE_LIMIT_MS - Date.now());
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  });
  chainTail = myTurn;
  await myTurn;
}

function noteCallEnded(): void {
  lastCallEndMs = Date.now();
}

function parseRetryDelaySeconds(err: unknown): number | null {
  const msg = (err as Error)?.message ?? "";
  const m = msg.match(/retry in (\d+(?:\.\d+)?)s/i);
  return m && m[1] ? parseFloat(m[1]) : null;
}

async function withRetry<T>(fn: () => Promise<T>, label: string): Promise<T> {
  const MAX_ATTEMPTS = 4;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await fn();
    } catch (e) {
      const msg = (e as Error).message ?? "";
      const isRateLimit = msg.includes("429") || msg.includes("rate limit");
      if (!isRateLimit || attempt === MAX_ATTEMPTS) throw e;
      const hinted = parseRetryDelaySeconds(e);
      const wait = hinted ? hinted * 1000 + 1000 : RATE_LIMIT_MS * attempt;
      stepError("llm", `[llm:${label}] 429, waiting ${(wait / 1000).toFixed(1)}s (attempt ${attempt})`);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw new Error("unreachable");
}

// OpenRouter models occasionally drop required JSON fields depending on the underlying provider. 
// We retry up to MAX_SCHEMA_RETRIES times, feeding the prior failure back as additional
// system context so the model fills in what it missed.
const MAX_SCHEMA_RETRIES = 2;

export async function structured<T extends ZodTypeAny>(args: {
  model: string;
  system: string;
  user: string;
  schema: T;
}): Promise<z.infer<T>> {
  const c = client();
  const jsonSchema = toJsonSchema(args.schema);

  const systemPromptWithSchema = `${args.system}\n\nYou must respond ONLY with valid JSON that strictly satisfies this schema:\n${JSON.stringify(jsonSchema, null, 2)}`;

  const finish = stepStart("llm", `model=${args.model}`);
  logSent("llm", {
    model: args.model,
    systemLen: `${systemPromptWithSchema.length} chars`,
    userLen: `${args.user.length} chars`,
  });

  let lastError: Error | null = null;
  let lastRawText: string | null = null;

  for (let attempt = 0; attempt <= MAX_SCHEMA_RETRIES; attempt++) {
    const messages: { role: "system" | "user"; content: string }[] = [
      { role: "system", content: systemPromptWithSchema },
      { role: "user", content: args.user },
    ];

    if (lastError && attempt > 0) {
      messages.push({
        role: "system",
        content: `Your previous response failed schema validation:\n${lastError.message}\n\nPrior raw output (truncated):\n${(lastRawText ?? "").slice(0, 400)}\n\nReply again with valid JSON. Include EVERY required field from the schema. No prose outside the JSON object.`,
      });
    }

    try {
      const text = await withRetry(async () => {
        await rateLimit();
        try {
          const response = await c.chat.completions.create({
            model: args.model,
            messages,
            temperature: 0,
            max_tokens: 6000,
            response_format: { type: "json_object" },
          });
          const t = response.choices[0]?.message?.content;
          if (!t) throw new Error("OpenRouter returned an empty response");
          return t;
        } finally {
          noteCallEnded();
        }
      }, args.model);

      lastRawText = text;

      let raw: unknown;
      try {
        raw = JSON.parse(text);
      } catch {
        lastError = new Error(`Model returned non-JSON: ${text.slice(0, 200)}`);
        continue;
      }

      const parsed = args.schema.parse(raw);
      logReceived("llm", parsed as Record<string, unknown>);
      if (attempt > 0) {
        stepError("llm", `Recovered after ${attempt} schema retr${attempt === 1 ? "y" : "ies"}`);
      }
      finish();
      return parsed;
    } catch (error) {
      lastError = error as Error;
    }
  }

  stepError("llm", `Error calling ${args.model} after ${MAX_SCHEMA_RETRIES + 1} attempts: ${lastError?.message}`);
  finish();
  throw lastError ?? new Error("structured() failed without specific error");
}