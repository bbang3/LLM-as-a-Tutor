"""paired_delta_constraint_effect.py

Compute paired Δ (treated − original) on the JSON produced by
``eval_constraint_effect.py`` (and optionally augmented with per-rollout
aggregates by ``augment_per_rollout_aggregates.py``).

Reads pre-computed per-rollout scalars directly — no recomputation from raw
3-D score tensors. Required fields per record (raises KeyError otherwise):

    rollout_scores_original_general
    rollout_scores_constraint_general

Usage::

    python analysis/scripts/paired_delta_constraint_effect.py \\
        --eval-json analysis/outputs/eval_constraint_effect_v14_8b_ep1.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# ─────────────────────────── hardcoded schema ──────────────────────────────

V1_SCHEMA = {
    "name": "v1 (append, shared rubric)",
    "score_original_key":  "rollout_scores_original_general",
    "score_treated_key":   "rollout_scores_constraint_general",
}


# ─────────────────────────── helpers ────────────────────────────────────────


def _extract_scalars(rec: dict, key: str) -> list[float]:
    if key not in rec:
        raise KeyError(
            f"missing per-rollout aggregate key '{key}' in record. "
            "Run analysis/scripts/augment_per_rollout_aggregates.py "
            "on the eval JSON first."
        )
    raw = rec[key] or []
    return [float(v) for v in raw if v is not None]


def _percentiles(xs: list[float], probs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
    if not xs:
        return [float("nan")] * len(probs)
    s = sorted(xs)
    n = len(s)
    return [s[max(0, min(n - 1, int(p * n)))] for p in probs]


def _summarize(label: str, xs: list[float]) -> str:
    n = len(xs)
    if n == 0:
        return f"{label:<14} n=0"
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    md = statistics.median(xs)
    p5, p25, _, p75, p95 = _percentiles(xs)
    return (
        f"{label:<14} n={n:<5}  mean={m:7.3f}  std={sd:6.3f}  median={md:6.2f}  "
        f"p5={p5:6.2f}  p25={p25:6.2f}  p75={p75:6.2f}  p95={p95:6.2f}"
    )


def _report_paired(label: str, deltas: list[float]) -> str:
    n = len(deltas)
    if n == 0:
        return f"{label:<22} n=0"
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 0 else 0.0
    t = m / sem if sem > 0 else 0.0
    md = statistics.median(deltas)
    p5, p25, _, p75, p95 = _percentiles(deltas)
    n_dn = sum(1 for x in deltas if x < 0)
    n_up = sum(1 for x in deltas if x > 0)
    n_eq = n - n_dn - n_up
    return (
        f"{label:<22} n={n:<5}  mean={m:+7.3f}  sd={sd:6.3f}  sem={sem:5.3f}  "
        f"t={t:+6.2f}  median={md:+6.3f}  p5={p5:+6.2f}  p25={p25:+6.2f}  "
        f"p75={p75:+6.2f}  p95={p95:+6.2f}  down/up/eq={n_dn}/{n_up}/{n_eq}"
    )


# ─────────────────────────── main ───────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--min-rollouts", type=int, default=2)
    ap.add_argument("--out-vectors", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {args.eval_json}")

    schema = V1_SCHEMA
    print(f"  schema: {schema['name']}")
    print(f"  original  : {schema['score_original_key']}")
    print(f"  treated   : {schema['score_treated_key']}")
    print(f"  min rollouts per condition: {args.min_rollouts}")

    rows = []
    n_skip_too_few = 0
    for rec in records:
        g_orig = _extract_scalars(rec, schema["score_original_key"])
        g_trt  = _extract_scalars(rec, schema["score_treated_key"])
        if len(g_orig) < args.min_rollouts or len(g_trt) < args.min_rollouts:
            n_skip_too_few += 1
            continue
        rows.append({
            "original_prompt": rec.get("original_prompt", ""),
            "n_rollouts_orig": len(g_orig),
            "n_rollouts_trt": len(g_trt),
            "orig_mean": statistics.mean(g_orig),
            "trt_mean": statistics.mean(g_trt),
            "orig_std": statistics.stdev(g_orig),
            "trt_std": statistics.stdev(g_trt),
            "orig_rollouts": g_orig,
            "trt_rollouts": g_trt,
        })

    n_paired = len(rows)
    print(f"  paired prompts: {n_paired} (skipped too-few-rollouts: {n_skip_too_few})")
    if n_paired == 0:
        return

    orig_pooled = [v for r in rows for v in r["orig_rollouts"]]
    trt_pooled  = [v for r in rows for v in r["trt_rollouts"]]
    orig_pp_mean = [r["orig_mean"] for r in rows]
    trt_pp_mean  = [r["trt_mean"]  for r in rows]
    orig_pp_std  = [r["orig_std"]  for r in rows]
    trt_pp_std   = [r["trt_std"]   for r in rows]

    print()
    print("=== Per-rollout pooled distribution ===")
    print(" ", _summarize("original", orig_pooled))
    print(" ", _summarize("treated",  trt_pooled))
    if orig_pooled and trt_pooled:
        d_m = statistics.mean(trt_pooled) - statistics.mean(orig_pooled)
        d_s = (statistics.stdev(trt_pooled) if len(trt_pooled) > 1 else 0) - (
              statistics.stdev(orig_pooled) if len(orig_pooled) > 1 else 0)
        print(f"  Δrollout-mean = {d_m:+.3f}    Δrollout-std = {d_s:+.3f}")

    print()
    print("=== Per-prompt mean distribution ===")
    print(" ", _summarize("original", orig_pp_mean))
    print(" ", _summarize("treated",  trt_pp_mean))

    print()
    print("=== Per-prompt within-prompt std distribution ===")
    print(" ", _summarize("original", orig_pp_std))
    print(" ", _summarize("treated",  trt_pp_std))

    pp_dm = [trt_pp_mean[i] - orig_pp_mean[i] for i in range(n_paired)]
    pp_ds = [trt_pp_std[i]  - orig_pp_std[i]  for i in range(n_paired)]

    print()
    print("=== Paired Δ per prompt (treated − original) ===")
    print(" ", _report_paired("Δ per-prompt mean", pp_dm))
    print(" ", _report_paired("Δ per-prompt std",  pp_ds))

    if args.out_vectors is not None:
        args.out_vectors.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_vectors, "w", encoding="utf-8") as f:
            json.dump({
                "orig_pooled": orig_pooled,
                "trt_pooled":  trt_pooled,
                "orig_pp_mean": orig_pp_mean,
                "trt_pp_mean":  trt_pp_mean,
                "orig_pp_std":  orig_pp_std,
                "trt_pp_std":   trt_pp_std,
                "pp_dmean":     pp_dm,
                "pp_dstd":      pp_ds,
            }, f)
        print(f"  saved vectors -> {args.out_vectors}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "original_prompt_prefix",
                "n_rollouts_orig", "n_rollouts_trt",
                "orig_mean", "trt_mean", "delta_mean",
                "orig_std",  "trt_std",  "delta_std",
            ])
            for r, dm, ds in zip(rows, pp_dm, pp_ds):
                w.writerow([
                    (r["original_prompt"] or "")[:60].replace("\n", " "),
                    r["n_rollouts_orig"], r["n_rollouts_trt"],
                    f"{r['orig_mean']:.4f}", f"{r['trt_mean']:.4f}", f"{dm:+.4f}",
                    f"{r['orig_std']:.4f}",  f"{r['trt_std']:.4f}",  f"{ds:+.4f}",
                ])
        print(f"  saved CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
