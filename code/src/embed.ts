// Local text embedding via @xenova/transformers (ONNX runtime in Node).
//
// We use BGE-small-en-v1.5 (384-dim, ~25 MB quantized). The model downloads
// on first use into node_modules/@xenova/transformers/.cache and is reused on
// every subsequent run. No API calls, no rate limits, fully deterministic.

import { pipeline, env, type FeatureExtractionPipeline } from "@xenova/transformers";

const MODEL_ID = "Xenova/bge-small-en-v1.5";

// Disable remote model fetching after the first download — keep the model in
// the local cache directory so repeated runs are fully offline.
env.allowLocalModels = true;
env.allowRemoteModels = true;

let _pipe: FeatureExtractionPipeline | null = null;

async function getPipeline(): Promise<FeatureExtractionPipeline> {
  if (_pipe) return _pipe;
  _pipe = (await pipeline("feature-extraction", MODEL_ID, {
    quantized: true,
  })) as FeatureExtractionPipeline;
  return _pipe;
}

export const modelId = (): string => MODEL_ID;

export async function embed(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const pipe = await getPipeline();
  // BGE expects a query prefix for retrieval queries; for documents we feed
  // raw text. We use raw text for both — symmetrical encoding still works
  // well, and the small quality loss isn't worth the prompt-engineering
  // complexity at this corpus scale.
  const result = await pipe(texts, { pooling: "mean", normalize: true });
  const dims = result.dims;
  const dim = dims[dims.length - 1] ?? 0;
  const flat = result.data as Float32Array;
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    out.push(Array.from(flat.slice(i * dim, (i + 1) * dim)));
  }
  return out;
}
