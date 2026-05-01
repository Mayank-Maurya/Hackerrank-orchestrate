// Reciprocal-rank fusion. Combines an arbitrary number of ranked lists into a
// single fused ranking, using the standard formula:
//
//     score(d) = Σ 1 / (k + rank_i(d))
//
// where rank_i(d) is the 1-based rank of document d in list i (or ∞ if absent).
// k=60 is the canonical default.

export interface RankedHit {
  idx: number;
  score: number;
}

export function rrf(
  rankings: RankedHit[][],
  opts: { k?: number; topK?: number } = {},
): RankedHit[] {
  const k = opts.k ?? 60;
  const topK = opts.topK ?? 20;
  const fused = new Map<number, number>();

  for (const list of rankings) {
    for (let rank = 0; rank < list.length; rank++) {
      const hit = list[rank];
      if (!hit) continue;
      const contrib = 1 / (k + rank + 1);
      fused.set(hit.idx, (fused.get(hit.idx) ?? 0) + contrib);
    }
  }

  const out: RankedHit[] = Array.from(fused, ([idx, score]) => ({ idx, score }));
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, topK);
}
