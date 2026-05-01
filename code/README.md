# Support Triage Agent

A terminal-based support triage agent built as a **typed state machine**. Each
ticket walks a discriminated-union pipeline of states, with explicit transitions
between them and per-stage retries on validation failure.

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
  gate. Hardcoded patterns + a small LLM safety classifier (`gemini-2.5-flash`).
- **Classify** — chooses `company`, `product_area` (from a closed list of corpus
  folders), `request_type`, and provisional `status`. One structured-output LLM
  call (`gemini-2.5-flash`).
- **Retrieve** — hybrid BM25 + embedding retrieval with reciprocal-rank fusion
  (RRF), scoped to the chosen company.
- **Draft** — grounded response + justification (`gemini-2.5-pro`).
- **Validate** — evidence check: every factual claim must be supported by a
  retrieved chunk. Unsupported claims trigger a retry with feedback, or escalation
  after 2 failures.

Provider: **Google Gemini** via the `@google/genai` SDK. Free tier covers the
full submission run plus dev iterations.

All retrieval, BM25, and RRF are hand-rolled — no LangChain.

## Setup

```bash
cd code
npm install
cp .env.example .env
# edit .env to add your OPENAI_API_KEY
```

## Running

```bash
# Run on sample_support_tickets.csv (development)
npm run start:sample

# Run on the real support_tickets.csv → writes support_tickets/output.csv
npm start
```

## Layout

```
code/
├── package.json
├── tsconfig.json
├── .env.example
├── README.md
└── src/
    ├── main.ts              # entry point
    ├── pipeline.ts          # state-machine runner
    ├── state.ts             # discriminated-union state types
    ├── types.ts             # shared types and constants
    ├── llm.ts               # OpenAI client + structured-output helper
    ├── csv.ts               # CSV read/write
    ├── corpus.ts            # data/ loader and chunking
    ├── retrieval/
    │   ├── bm25.ts          # hand-rolled BM25
    │   ├── embeddings.ts    # cosine over OpenAI embeddings
    │   └── rrf.ts           # reciprocal-rank fusion
    └── nodes/
        ├── triage.ts
        ├── classify.ts
        ├── retrieve.ts
        ├── draft.ts
        └── validate.ts
```

## Determinism

- All LLM calls use `temperature=0` and the seed in `OPENAI_SEED`.
- Embeddings are deterministic by API contract.
- BM25 and RRF are pure functions over the indexed corpus.
- The corpus index is cached on disk under `data/index/` and `data/embeddings/`
  (both gitignored) and rebuilt only when the corpus changes.

## Submission

Zip the `code/` directory excluding `node_modules/` and `dist/`.
