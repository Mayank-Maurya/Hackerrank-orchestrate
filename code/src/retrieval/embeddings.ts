// Embedding retrieval. Cosine similarity over local BGE-small embeddings
// (see ../embed.ts), with on-disk caching keyed by a hash of
// (corpus content + model) so we only pay to embed once per corpus version.

import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { embed, modelId } from "../embed.js";
import type { Chunk } from "../types.js";

const CACHE_DIR = new URL("../../../data/embeddings/", import.meta.url).pathname;

export interface EmbeddingIndex {
  chunks: Chunk[];
  vectors: Float32Array[]; // aligned with chunks
  dim: number;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export function embeddingSearch(
  index: EmbeddingIndex,
  queryVec: Float32Array,
  opts: { topK?: number; companyFilter?: Chunk["company"] } = {},
): { idx: number; score: number }[] {
  const topK = opts.topK ?? 20;
  const scored: { idx: number; score: number }[] = [];
  for (let i = 0; i < index.chunks.length; i++) {
    const c = index.chunks[i];
    if (!c) continue;
    if (opts.companyFilter && c.company !== opts.companyFilter) continue;
    const v = index.vectors[i];
    if (!v) continue;
    scored.push({ idx: i, score: cosine(queryVec, v) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// ---- Cache layout ------------------------------------------------------------
//   data/embeddings/<hash>.json  : { model, dim, count, paths }
//   data/embeddings/<hash>.bin   : raw Float32 LE, count * dim values
//
// `hash` = SHA256(model | dim-placeholder | sorted (path|content) entries).
// Dim is unknown until first embed succeeds; we recompute the hash without
// dim and rely on (model + content) being sufficient.

interface CacheManifest {
  model: string;
  dim: number;
  count: number;
  paths: string[];
}

function corpusHash(chunks: Chunk[], model: string): string {
  const h = createHash("sha256");
  h.update(`model:${model}\n`);
  const sorted = [...chunks].sort((a, b) => a.path.localeCompare(b.path));
  for (const c of sorted) {
    h.update(c.path);
    h.update("\0");
    h.update(c.content);
    h.update("\0");
  }
  return h.digest("hex").slice(0, 16);
}

function chunkText(c: Chunk): string {
  // Keep title prominent; truncate to stay well under text-embedding-004's
  // 2048-token input limit (~7-8k chars is safe).
  const head = `${c.title}\n\n`;
  const body = c.content.slice(0, 7000 - head.length);
  return head + body;
}

async function loadFromCache(hash: string, chunks: Chunk[]): Promise<EmbeddingIndex | null> {
  const manifestPath = join(CACHE_DIR, `${hash}.json`);
  const binPath = join(CACHE_DIR, `${hash}.bin`);
  try {
    await stat(manifestPath);
    await stat(binPath);
  } catch {
    return null;
  }

  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as CacheManifest;
  if (manifest.count !== chunks.length) return null;

  // Verify path alignment with current corpus order.
  const sortedChunks = [...chunks].sort((a, b) => a.path.localeCompare(b.path));
  for (let i = 0; i < sortedChunks.length; i++) {
    if (sortedChunks[i]!.path !== manifest.paths[i]) return null;
  }

  const buf = await readFile(binPath);
  const view = new Float32Array(
    buf.buffer,
    buf.byteOffset,
    buf.byteLength / 4,
  );

  // Index keyed by path so we can re-assemble vectors aligned to `chunks`
  // (whose order may differ from the cached sort order).
  const byPath = new Map<string, Float32Array>();
  for (let i = 0; i < manifest.count; i++) {
    const slice = view.slice(i * manifest.dim, (i + 1) * manifest.dim);
    byPath.set(manifest.paths[i]!, slice);
  }

  const vectors: Float32Array[] = [];
  for (const c of chunks) {
    const v = byPath.get(c.path);
    if (!v) return null;
    vectors.push(v);
  }

  return { chunks, vectors, dim: manifest.dim };
}

async function saveToCache(
  hash: string,
  manifest: CacheManifest,
  vectorsByPath: Map<string, Float32Array>,
): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(
    join(CACHE_DIR, `${hash}.json`),
    JSON.stringify(manifest, null, 2),
  );
  // Write vectors in `manifest.paths` order so loadFromCache can reconstruct.
  const total = manifest.count * manifest.dim * 4;
  const buf = Buffer.alloc(total);
  let offset = 0;
  for (const p of manifest.paths) {
    const v = vectorsByPath.get(p)!;
    for (let i = 0; i < v.length; i++) {
      buf.writeFloatLE(v[i]!, offset);
      offset += 4;
    }
  }
  await writeFile(join(CACHE_DIR, `${hash}.bin`), buf);
}

const EMBED_BATCH = 32; // contents per local-ONNX call (CPU sweet spot)

export async function buildEmbeddingIndex(chunks: Chunk[]): Promise<EmbeddingIndex> {
  const model = modelId();
  const hash = corpusHash(chunks, model);

  const cached = await loadFromCache(hash, chunks);
  if (cached) {
    console.log(`[embed] cache hit (${cached.vectors.length} vectors, dim=${cached.dim})`);
    return cached;
  }

  console.log(`[embed] cache miss, embedding ${chunks.length} chunks via ${model}`);
  const vectorsByPath = new Map<string, Float32Array>();
  let dim = 0;

  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    const batch = chunks.slice(i, i + EMBED_BATCH);
    const texts = batch.map(chunkText);
    const vecs = await embed(texts);
    if (vecs.length !== batch.length) {
      throw new Error(
        `Embedding batch returned ${vecs.length} vectors for ${batch.length} inputs`,
      );
    }
    for (let j = 0; j < batch.length; j++) {
      const arr = Float32Array.from(vecs[j]!);
      if (dim === 0) dim = arr.length;
      vectorsByPath.set(batch[j]!.path, arr);
    }
    console.log(`[embed]   ${Math.min(i + EMBED_BATCH, chunks.length)}/${chunks.length}`);
  }

  // Build manifest in sorted order for reproducible cache files.
  const sortedPaths = [...chunks].map((c) => c.path).sort();
  const manifest: CacheManifest = {
    model,
    dim,
    count: chunks.length,
    paths: sortedPaths,
  };
  await saveToCache(hash, manifest, vectorsByPath);

  const vectors = chunks.map((c) => vectorsByPath.get(c.path)!);
  return { chunks, vectors, dim };
}
