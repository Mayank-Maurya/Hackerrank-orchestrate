// Centralized debug logger for pipeline visibility.
// Prints colorized, timestamped step-level logs to stderr so they don't
// interfere with stdout data output.

const COLORS = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  bold: "\x1b[1m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  magenta: "\x1b[35m",
  red: "\x1b[31m",
  blue: "\x1b[34m",
  white: "\x1b[37m",
  gray: "\x1b[90m",
} as const;

const STEP_COLORS: Record<string, string> = {
  triage: COLORS.cyan,
  classify: COLORS.yellow,
  retrieve: COLORS.blue,
  draft: COLORS.magenta,
  validate: COLORS.green,
  pipeline: COLORS.white,
  llm: COLORS.red,
  corpus: COLORS.cyan,
  embed: COLORS.blue,
};

function timestamp(): string {
  const d = new Date();
  return d.toISOString().replace("T", " ").replace("Z", "");
}

function truncate(s: string, max = 200): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + `… (${s.length} chars)`;
}

function elapsed(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Log the start of a named step. Returns a finish function that logs
 *  completion + elapsed time when called. */
export function stepStart(step: string, detail?: string): () => void {
  const color = STEP_COLORS[step] ?? COLORS.white;
  const detailStr = detail ? ` ${COLORS.dim}${truncate(detail)}${COLORS.reset}` : "";
  console.error(
    `${COLORS.gray}${timestamp()}${COLORS.reset} ${color}${COLORS.bold}▶ ${step.toUpperCase()}${COLORS.reset}${detailStr}`,
  );
  const t0 = performance.now();
  return () => {
    const dt = performance.now() - t0;
    console.error(
      `${COLORS.gray}${timestamp()}${COLORS.reset} ${color}${COLORS.bold}✓ ${step.toUpperCase()}${COLORS.reset} ${COLORS.dim}(${elapsed(dt)})${COLORS.reset}`,
    );
  };
}

/** Log a key→value detail under the current step. */
export function stepDetail(step: string, key: string, value: string): void {
  const color = STEP_COLORS[step] ?? COLORS.white;
  console.error(
    `${COLORS.gray}${timestamp()}${COLORS.reset} ${color}  ├─ ${key}:${COLORS.reset} ${COLORS.dim}${truncate(value, 300)}${COLORS.reset}`,
  );
}

/** Log a warning. */
export function stepWarn(step: string, msg: string): void {
  const color = STEP_COLORS[step] ?? COLORS.white;
  console.error(
    `${COLORS.gray}${timestamp()}${COLORS.reset} ${color}  ⚠  ${msg}${COLORS.reset}`,
  );
}

/** Log an error. */
export function stepError(step: string, msg: string): void {
  console.error(
    `${COLORS.gray}${timestamp()}${COLORS.reset} ${COLORS.red}${COLORS.bold}  ✗ [${step}] ${msg}${COLORS.reset}`,
  );
}

/** Convenience: log sent payload summary. */
export function logSent(step: string, payload: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(payload)) {
    const display =
      typeof v === "string"
        ? truncate(v, 150)
        : JSON.stringify(v)?.slice(0, 150) ?? "undefined";
    stepDetail(step, `→ ${k}`, display);
  }
}

/** Convenience: log received payload summary. */
export function logReceived(step: string, payload: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(payload)) {
    const display =
      typeof v === "string"
        ? truncate(v, 150)
        : JSON.stringify(v)?.slice(0, 150) ?? "undefined";
    stepDetail(step, `← ${k}`, display);
  }
}
