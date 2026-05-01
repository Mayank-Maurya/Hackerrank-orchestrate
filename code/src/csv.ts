// CSV read/write. Input headers vary in case ("Issue" vs "issue"); output uses
// the lowercase_underscore header that already exists in support_tickets/output.csv.

import { readFile, writeFile } from "node:fs/promises";
import { parse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";

import type { Company, InputRow, OutputRow } from "./types.js";
import { COMPANY } from "./types.js";

// Map any of the variant input column headers to our canonical lowercase keys.
const HEADER_MAP: Record<string, keyof InputRow> = {
  issue: "issue",
  Issue: "issue",
  subject: "subject",
  Subject: "subject",
  company: "company",
  Company: "company",
};

function normalizeCompany(raw: string): Company {
  const t = raw.trim();
  for (const c of COMPANY) {
    if (t.toLowerCase() === c.toLowerCase()) return c;
  }
  // Default unknown-company tickets to "None" so the triage stage handles them.
  return "None";
}

export async function readInputCsv(path: string): Promise<InputRow[]> {
  const buf = await readFile(path);
  const records: Record<string, string>[] = parse(buf, {
    columns: (headers: string[]) => headers.map((h) => HEADER_MAP[h] ?? h),
    skip_empty_lines: true,
    trim: true,
  });
  return records.map((r) => ({
    issue: (r.issue ?? "").trim(),
    subject: (r.subject ?? "").trim(),
    company: normalizeCompany(r.company ?? "None"),
  }));
}

export async function writeOutputCsv(
  path: string,
  rows: OutputRow[],
): Promise<void> {
  const csv = stringify(rows, {
    header: true,
    columns: [
      "issue",
      "subject",
      "company",
      "response",
      "product_area",
      "status",
      "request_type",
      "justification",
    ],
    quoted_string: true,
  });
  await writeFile(path, csv, "utf8");
}
