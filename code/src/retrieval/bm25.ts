// Hand-rolled BM25 (Okapi).
//
// Stub: tokenize, indexing, and scoring scaffolding are in place. The real
// math (idf, length norm, score function) is filled in below; the index
// builder is intentionally minimal so we can unit-check it against simple
// inputs before wiring it up.

import type { Chunk } from "../types.js";

const STOPWORDS = new Set(
  "a an and are as at be by for from has have he i in is it its of on or that the their this to was were will with you your".split(
    " ",
  ),
);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[`*_~>#]/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

export interface Bm25Index {
  // Per-document tokens, kept aligned with `chunks`.
  docs: string[][];
  chunks: Chunk[];
  // Document frequency for each term.
  df: Map<string, number>;
  // Average document length (in tokens).
  avgdl: number;
  // Total number of documents.
  N: number;
}

export function buildBm25Index(chunks: Chunk[]): Bm25Index {
  const docs = chunks.map((c) => tokenize(`${c.title}\n${c.content}`));
  const df = new Map<string, number>();
  for (const tokens of docs) {
    const seen = new Set(tokens);
    for (const t of seen) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const totalLen = docs.reduce((s, d) => s + d.length, 0);
  const avgdl = docs.length === 0 ? 0 : totalLen / docs.length;
  return { docs, chunks, df, avgdl, N: docs.length };
}

const K1 = 1.5;
const B = 0.75;

function idf(N: number, df: number): number {
  // Robertson-Sparck Jones IDF, clamped to non-negative.
  return Math.max(0, Math.log(1 + (N - df + 0.5) / (df + 0.5)));
}

export function bm25Search(
  index: Bm25Index,
  query: string,
  opts: { topK?: number; companyFilter?: Chunk["company"] } = {},
): { idx: number; score: number }[] {
  const qTokens = tokenize(query);
  const topK = opts.topK ?? 20;

  const scored: { idx: number; score: number }[] = [];
  for (let i = 0; i < index.docs.length; i++) {
    const chunk = index.chunks[i];
    if (!chunk) continue;
    if (opts.companyFilter && chunk.company !== opts.companyFilter) continue;

    const doc = index.docs[i];
    if (!doc) continue;

    // Term-frequency map for this document.
    const tf = new Map<string, number>();
    for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);

    let score = 0;
    const dl = doc.length;
    for (const q of qTokens) {
      const f = tf.get(q);
      if (!f) continue;
      const dfq = index.df.get(q) ?? 0;
      const num = f * (K1 + 1);
      const den = f + K1 * (1 - B + (B * dl) / (index.avgdl || 1));
      score += idf(index.N, dfq) * (num / den);
    }
    if (score > 0) scored.push({ idx: i, score });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
