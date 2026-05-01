# Support Triage Agent

A terminal-based support triage agent built as a **typed state machine** in
TypeScript. Each ticket walks a discriminated-union pipeline of states with
explicit transitions and per-stage retries on validation failure.

## Architecture

```
RawTicket ─► Triaged ─► Classified ─► Retrieved ─► Drafted ─► Validated ─► Final
                │            │                                    │
                ▼            ▼                                    ▼ (retry)
              Final        Final                                Drafted
            (escalated)  (escalated/                           (≤2 retries)
                          invalid)
```

- **Triage** — fast prompt-injection / sensitive-intent / greeting / out-of-scope
  gate. Hardcoded regex (tier-1) + a small LLM safety classifier (tier-2).
- **Classify** — chooses `company`, `product_area` (constrained to a closed
  list of corpus subfolders), `request_type`, and provisional `status`. One
  structured-output LLM call.
- **Retrieve** — hybrid BM25 + local embedding (BGE-small via ONNX) retrieval
  with reciprocal-rank fusion (RRF), scoped to the chosen company.
- **Draft** — grounded response + justification with strict "use only the
  provided chunks" system prompt.
- **Validate** — evidence check: every factual claim must be supported by a
  retrieved chunk. Unsupported claims trigger a retry with feedback, or
  escalation after 2 failures.

Schema-validation failures from the LLM trigger up to 2 automatic retries
inside the `structured()` helper, feeding the prior failure message back as
additional system context.

All retrieval, BM25, RRF, and corpus chunking are hand-rolled — no LangChain.

## Provider

The agent uses an OpenAI-compatible client pointed at **OpenRouter**. Default
model is `google/gemini-2.5-flash` for every node (triage, classify, draft,
validate). Embeddings run **fully locally** via
[`@xenova/transformers`](https://www.npmjs.com/package/@xenova/transformers)
with `Xenova/bge-small-en-v1.5` — no API call, zero rate limit, 384-dim,
~25 MB ONNX model that downloads once and caches.

## Setup

```bash
cd code
npm install
cp .env.example .env
# edit code/.env to add your key:
#   OPEN_ROUTER_KEY=sk-or-v1-...
```

Get an OpenRouter key (free credits on signup) at
https://openrouter.ai/keys. The default model `google/gemini-2.5-flash` is
available on the free tier of OpenRouter as well as paid.

`.env` is gitignored. Never put a real key into `.env.example`.

## Running

```bash
# Dev: run on sample_support_tickets.csv → sample_output.csv
npm run start:sample

# Submission: run on support_tickets.csv → output.csv
npm start
```

The first run on a new corpus builds the local embedding index (~3,300 chunks,
~1–2 min on CPU) and caches it under `data/embeddings/<hash>.{bin,json}`.
Subsequent runs hit the cache instantly. The cache invalidates automatically
when corpus content changes.

Per-step debug logs (timestamped, colorized, with elapsed times) are emitted
to **stderr** so they don't pollute output. Pipe stderr to a file if you want
a clean log:

```bash
npm run start:sample 2> /tmp/run.log
```

Total run time on the 28-ticket submission set is roughly **5–10 min** with
the embedding cache warm, dominated by sequential LLM latency.

## Layout

```
code/
├── package.json
├── tsconfig.json
├── .env.example
├── README.md
└── src/
    ├── main.ts              # entry point, CLI flags, CSV in/out
    ├── pipeline.ts          # state-machine runner (exhaustive switch)
    ├── state.ts             # discriminated-union state types
    ├── types.ts             # closed-set values (Status, RequestType, Company)
    ├── llm.ts               # OpenAI-compatible client + structured-output helper
    ├── embed.ts             # local BGE-small embedding via @xenova/transformers
    ├── csv.ts               # CSV read/write with header normalization
    ├── corpus.ts            # data/ loader + heading-based chunking
    ├── logger.ts            # colorized step-level debug logger (stderr)
    ├── retrieval/
    │   ├── bm25.ts          # hand-rolled Okapi BM25
    │   ├── embeddings.ts    # cosine over Float32Array, on-disk cache
    │   └── rrf.ts           # reciprocal-rank fusion
    └── nodes/
        ├── triage.ts        # tier-1 regex + tier-2 LLM safety classifier
        ├── classify.ts      # company / product_area / request_type / status
        ├── retrieve.ts      # hybrid BM25 + embeddings + RRF
        ├── draft.ts         # grounded response generator
        └── validate.ts      # evidence-based fact-check + retry trigger
```

## Determinism

- All LLM calls use `temperature=0`.
- BGE-small embeddings are deterministic (ONNX, fixed weights).
- BM25 and RRF are pure functions over the indexed corpus.
- The corpus index is cached on disk under `data/embeddings/` (gitignored)
  and rebuilt only when corpus content changes (SHA256 over chunks + model).

## Submission

Zip `code/` excluding `node_modules/`, `dist/`, and `.env`:

```bash
zip -r submission.zip code -x 'code/node_modules/*' 'code/dist/*' 'code/.env'
```

Upload the zip + `support_tickets/output.csv` + `~/hackerrank_orchestrate/log.txt`
on the HackerRank Community Platform.
