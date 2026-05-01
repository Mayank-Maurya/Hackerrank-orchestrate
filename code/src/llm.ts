// Local LiteLLM client + structured-output helper.
//
// We use the `openai` SDK pointing to a local LiteLLM endpoint with `response_format: { type: "json_object" }`
// and a JSON Schema (converted from a Zod schema) injected into the system prompt for structured output. 
// The SDK returns raw JSON text; we then parse with Zod for runtime validation.

import OpenAI from "openai";
import type { z, ZodTypeAny } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

import { stepStart, logSent, logReceived, stepError } from "./logger.js";

let _client: OpenAI | null = null;

export function client(): OpenAI {
  if (_client) return _client;

  // LiteLLM requires an API key string, but it can be a dummy value if auth isn't configured
  const apiKey = process.env.LITELLM_API_KEY || "password";
  const baseURL = process.env.LITELLM_BASE_URL || "http://192.168.1.8:4000/v1";

  _client = new OpenAI({
    apiKey,
    baseURL
  });

  return _client;
}

export const MODELS = {
  // Update these defaults to match the model string in your LiteLLM config
  triage: () => process.env.LOCAL_TRIAGE_MODEL ?? "ollama/qwen2.5:14b-instruct",
  classify: () => process.env.LOCAL_CLASSIFY_MODEL ?? "ollama/qwen2.5:14b-instruct",
  draft: () => process.env.LOCAL_DRAFT_MODEL ?? "ollama/qwen2.5:14b-instruct",
  validate: () => process.env.LOCAL_VALIDATE_MODEL ?? "ollama/qwen2.5:14b-instruct",
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

// Local inference handles sequential processing best. 
// We serialize all calls through a single gate to prevent thrashing the VRAM,
// but we remove the arbitrary 5-second wait since local hardware dictates the speed.
let chainTail: Promise<void> = Promise.resolve();

async function serializeExecution(): Promise<void> {
  const myTurn = chainTail.then(() => Promise.resolve());
  chainTail = myTurn;
  await myTurn;
}

export async function structured<T extends ZodTypeAny>(args: {
  model: string;
  system: string;
  user: string;
  schema: T;
}): Promise<z.infer<T>> {
  const c = client();
  const jsonSchema = toJsonSchema(args.schema);

  // Local models perform best at JSON tasks when the schema is explicitly handed to them
  // in the system prompt alongside setting the json_object flag.
  const systemPromptWithSchema = `${args.system}\n\nYou must respond ONLY with valid JSON that strictly satisfies this schema:\n${JSON.stringify(jsonSchema, null, 2)}`;

  await serializeExecution();

  const finish = stepStart("llm", `model=${args.model}`);
  logSent("llm", {
    model: args.model,
    systemLen: `${systemPromptWithSchema.length} chars`,
    userLen: `${args.user.length} chars`,
  });

  try {
    const response = await c.chat.completions.create({
      model: args.model,
      messages: [
        { role: "system", content: systemPromptWithSchema },
        { role: "user", content: args.user }
      ],

      temperature: 0,
      response_format: { type: "json_object" },
    });


    const text = response.choices[0]?.message?.content;

    if (!text) {
      throw new Error("Local model returned an empty response");
    }

    let raw: unknown;
    try {
      raw = JSON.parse(text);
    } catch {
      throw new Error(`Model returned non-JSON: ${text.slice(0, 200)}`);
    }
    const parsed = args.schema.parse(raw);
    logReceived("llm", parsed as Record<string, unknown>);
    finish();
    return parsed;

  } catch (error) {
    stepError("llm", `Error calling ${args.model}: ${(error as Error).message}`);
    finish();
    throw error;
  }
}