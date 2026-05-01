// Corpus loader. Walks data/ and splits each markdown file into chunks at
// heading boundaries (##, ###, ####). Each section becomes its own Chunk,
// improving retrieval precision over one-chunk-per-file.

import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, sep } from "node:path";

import type { Chunk } from "./types.js";

const DATA_ROOT = new URL("../../data/", import.meta.url).pathname;

/** Minimum character length for a chunk. Shorter sections are merged into the
 *  previous chunk to avoid noisy fragments. */
const MIN_CHUNK_LENGTH = 80;

const COMPANY_BY_TOP_FOLDER: Record<string, Chunk["company"]> = {
  hackerrank: "HackerRank",
  claude: "Claude",
  visa: "Visa",
};

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

function deriveH1Title(content: string, fallback: string): string {
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

/** Strip YAML frontmatter (--- ... ---) from the beginning of content. */
function stripFrontmatter(raw: string): string {
  const m = raw.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
  return m ? raw.slice(m[0].length) : raw;
}

// ---------------------------------------------------------------------------
// Heading-based chunking
// ---------------------------------------------------------------------------

interface RawSection {
  heading: string; // The heading text (without the leading `#` markers).
  body: string;    // All text under this heading until the next heading.
}

/**
 * Split markdown content into sections at heading boundaries.
 * Headings of level 2–6 (i.e. `##` through `######`) mark section starts.
 * The H1 title and any text before the first sub-heading form the preamble.
 */
function splitByHeading(content: string): RawSection[] {
  const body = stripFrontmatter(content);
  const lines = body.split(/\r?\n/);

  const sections: RawSection[] = [];
  let currentHeading = ""; // Will be overwritten by the H1 or preamble
  let currentLines: string[] = [];

  for (const line of lines) {
    // Match ## through ###### (but NOT single # which is the page title)
    const headingMatch = line.match(/^(#{2,6})\s+(.+)$/);
    if (headingMatch) {
      // Flush the previous section
      if (currentLines.length > 0 || currentHeading) {
        sections.push({
          heading: currentHeading,
          body: currentLines.join("\n").trim(),
        });
      }
      // Clean heading text: strip bold markers like **...**
      currentHeading = (headingMatch[2] ?? "").replace(/\*\*/g, "").trim();
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }

  // Flush final section
  if (currentLines.length > 0 || currentHeading) {
    sections.push({
      heading: currentHeading,
      body: currentLines.join("\n").trim(),
    });
  }

  return sections;
}

/**
 * Convert raw sections into Chunk objects. Tiny trailing sections are merged
 * into the previous chunk. The preamble chunk uses the H1 as its title.
 */
function chunkFile(
  content: string,
  rel: string,
  company: Chunk["company"],
  productArea: string,
): Chunk[] {
  const h1 = deriveH1Title(content, rel);
  const sections = splitByHeading(content);

  if (sections.length === 0) {
    // Degenerate case: empty file → single chunk with full content
    return [{ path: rel, company, productArea, title: h1, content }];
  }

  // Build preliminary chunks
  const preliminary: { title: string; body: string }[] = [];
  for (const sec of sections) {
    const title = sec.heading || h1;
    // Prefix section body with the heading for context when retrieved
    const body = sec.heading
      ? `## ${sec.heading}\n\n${sec.body}`
      : sec.body;
    preliminary.push({ title, body });
  }

  // Merge tiny chunks into the previous one
  const merged: { title: string; body: string }[] = [];
  for (const p of preliminary) {
    if (
      merged.length > 0 &&
      p.body.length < MIN_CHUNK_LENGTH
    ) {
      merged[merged.length - 1]!.body += "\n\n" + p.body;
    } else {
      merged.push({ ...p });
    }
  }

  // If everything got merged into one, just return the full doc
  if (merged.length <= 1) {
    return [{ path: rel, company, productArea, title: h1, content: stripFrontmatter(content) }];
  }

  return merged.map((m, i) => ({
    path: `${rel}#${i}`,
    company,
    productArea,
    title: m.title,
    content: m.body,
  }));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function loadCorpus(): Promise<Chunk[]> {
  const chunks: Chunk[] = [];
  for (const top of Object.keys(COMPANY_BY_TOP_FOLDER)) {
    const dir = join(DATA_ROOT, top);
    const files = await walkMarkdown(dir);
    for (const f of files) {
      const rel = relative(DATA_ROOT, f);
      const content = await readFile(f, "utf8");
      const productArea = deriveProductArea(rel);
      const company = COMPANY_BY_TOP_FOLDER[top];
      if (!company) continue;
      chunks.push(...chunkFile(content, rel, company, productArea));
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
