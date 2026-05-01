// Corpus loader. Walks data/ and produces one Chunk per markdown file. We can
// upgrade to heading-based chunking later; for the file sizes in this corpus
// (avg ~10 KB) one chunk per file is fine and keeps retrieval simple.

import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, sep } from "node:path";

import type { Chunk } from "./types.js";

const DATA_ROOT = new URL("../../data/", import.meta.url).pathname;

const COMPANY_BY_TOP_FOLDER: Record<string, Chunk["company"]> = {
  hackerrank: "HackerRank",
  claude: "Claude",
  visa: "Visa",
};

async function walkMarkdown(dir: string): Promise<string[]> {
  const out: string[] = [];
  const entries = await readdir(dir);
  for (const entry of entries) {
    const full = join(dir, entry);
    const s = await stat(full);
    if (s.isDirectory()) {
      out.push(...(await walkMarkdown(full)));
    } else if (entry.endsWith(".md")) {
      out.push(full);
    }
  }
  return out;
}

function deriveTitle(content: string, fallback: string): string {
  const m = content.match(/^#\s+(.+)$/m);
  return (m?.[1] ?? fallback).trim();
}

function deriveProductArea(relPath: string): string {
  // relPath: "claude/privacy-and-legal/12345-foo.md"
  // productArea: the second-from-top folder if present, else the top folder.
  const parts = relPath.split(sep);
  if (parts.length >= 3) {
    const second = parts[1];
    if (second) return second.replace(/-/g, "_");
  }
  return (parts[0] ?? "general").replace(/-/g, "_");
}

export async function loadCorpus(): Promise<Chunk[]> {
  const chunks: Chunk[] = [];
  for (const top of Object.keys(COMPANY_BY_TOP_FOLDER)) {
    const dir = join(DATA_ROOT, top);
    const files = await walkMarkdown(dir);
    for (const f of files) {
      const rel = relative(DATA_ROOT, f);
      const content = await readFile(f, "utf8");
      const title = deriveTitle(content, rel);
      const productArea = deriveProductArea(rel);
      const company = COMPANY_BY_TOP_FOLDER[top];
      if (!company) continue;
      chunks.push({ path: rel, company, productArea, title, content });
    }
  }
  return chunks;
}

// Closed list of product_area values per company, derived from the first level
// inside each company folder. Useful for the Classify node which constrains
// LLM output to known values.
export function productAreasByCompany(chunks: Chunk[]): Record<string, string[]> {
  const out: Record<string, Set<string>> = {};
  for (const ch of chunks) {
    (out[ch.company] ??= new Set()).add(ch.productArea);
  }
  return Object.fromEntries(
    Object.entries(out).map(([k, v]) => [k, Array.from(v).sort()]),
  );
}
