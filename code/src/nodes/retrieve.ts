// Retrieve node — hybrid BM25 + embeddings with reciprocal-rank fusion,
// scoped to the company chosen at classification.
//
// The corpus + indices are loaded once and threaded through the pipeline as
// `ctx`. The scaffold returns empty chunks so downstream nodes still work.

import type { Bm25Index } from "../retrieval/bm25.js";
import type { EmbeddingIndex } from "../retrieval/embeddings.js";
import { bm25Search } from "../retrieval/bm25.js";
import { embeddingSearch } from "../retrieval/embeddings.js";
import { rrf } from "../retrieval/rrf.js";
import type { Classified, Retrieved } from "../state.js";
import type { Chunk, ScoredChunk } from "../types.js";

export interface RetrievalContext {
  bm25: Bm25Index;
  embeddings: EmbeddingIndex;
  // Aligned chunk array used by both indices.
  chunks: Chunk[];
  // Embed a single query string. Wired in main.ts to llm.embed.
  embedQuery: (text: string) => Promise<Float32Array>;
}

const TOP_K = 6;

export async function retrieve(
  c: Classified,
  ctx: RetrievalContext,
): Promise<Retrieved> {
  const company =
    c.classification.company === "Unknown"
      ? undefined
      : c.classification.company;

  const query = `${c.raw.input.subject}\n${c.raw.input.issue}`;
  const bm25Hits = bm25Search(ctx.bm25, query, {
    topK: TOP_K * 3,
    companyFilter: company,
  });

  let embedHits: { idx: number; score: number }[] = [];
  if (ctx.embeddings.vectors.length > 0) {
    const qVec = await ctx.embedQuery(query);
    embedHits = embeddingSearch(ctx.embeddings, qVec, {
      topK: TOP_K * 3,
      companyFilter: company,
    });
  }

  const fused = rrf([bm25Hits, embedHits], { topK: TOP_K });

  const chunks: ScoredChunk[] = [];
  for (const hit of fused) {
    const ch = ctx.chunks[hit.idx];
    if (!ch) continue;
    chunks.push({ ...ch, score: hit.score });
  }

  return {
    kind: "retrieved",
    raw: c.raw,
    flags: c.flags,
    classification: c.classification,
    chunks,
  };
}
