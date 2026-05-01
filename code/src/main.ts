// Entry point. Reads tickets, runs the state machine on each, writes the
// output CSV. Run with:
//
//   npm start                # support_tickets.csv -> output.csv
//   npm run start:sample     # sample_support_tickets.csv -> sample_output.csv

import "dotenv/config";
import { resolve } from "node:path";

import { readInputCsv, writeOutputCsv } from "./csv.js";
import { loadCorpus, productAreasByCompany } from "./corpus.js";
import { buildBm25Index } from "./retrieval/bm25.js";
import { buildEmbeddingIndex } from "./retrieval/embeddings.js";
import type { RawTicket } from "./state.js";
import { runPipeline } from "./pipeline.js";
import type { OutputRow } from "./types.js";
import { embed } from "./embed.js";

const REPO_ROOT = resolve(new URL("../../", import.meta.url).pathname);
const TICKETS_DIR = resolve(REPO_ROOT, "support_tickets");

interface RunPaths {
  input: string;
  output: string;
}

function pickPaths(): RunPaths {
  const useSample = process.argv.includes("--sample");
  if (useSample) {
    return {
      input: resolve(TICKETS_DIR, "sample_support_tickets.csv"),
      output: resolve(TICKETS_DIR, "sample_output.csv"),
    };
  }
  return {
    input: resolve(TICKETS_DIR, "support_tickets.csv"),
    output: resolve(TICKETS_DIR, "output.csv"),
  };
}

async function main() {
  const paths = pickPaths();
  console.log(`[main] reading ${paths.input}`);
  const inputs = await readInputCsv(paths.input);
  console.log(`[main] ${inputs.length} tickets loaded`);
  console.log(`[main] first ticket: `, inputs[0]);

  console.log(`[main] loading corpus`);
  const corpus = await loadCorpus();
  console.log(`[main] ${corpus.length} corpus chunks`);

  console.log(`[main] building BM25 index`);
  const bm25 = buildBm25Index(corpus);

  console.log(`[main] building embedding index`);
  const embeddings = await buildEmbeddingIndex(corpus);

  const ctx = {
    retrieval: {
      bm25,
      embeddings,
      chunks: corpus,
      embedQuery: async (text: string): Promise<Float32Array> => {
        const [vec] = await embed([text]);
        return Float32Array.from(vec ?? []);
      },
    },
    classify: {
      productAreasByCompany: productAreasByCompany(corpus),
    },
  };

  const outputs: OutputRow[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i]!;
    console.log(`[main] (${i + 1}/${inputs.length}) ${input.subject || input.issue.slice(0, 60)}`);
    const initial: RawTicket = { kind: "raw", input };
    const final = await runPipeline(initial, ctx);
    outputs.push({
      issue: final.input.issue,
      subject: final.input.subject,
      company: final.input.company,
      response: final.response,
      product_area: final.productArea,
      status: final.status,
      request_type: final.requestType,
      justification: final.justification,
    });
  }

  console.log(`[main] writing ${paths.output}`);
  await writeOutputCsv(paths.output, outputs);
  console.log(`[main] done`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
