"""augment_per_rollout_aggregates.py

Add per-rollout weighted-mean aggregates to an existing eval JSON in place
(or to a new path with --out). Idempotent — re-running just overwrites the
aggregate fields.

For each record, computes (subject to existence of corresponding raw score
field):
    rollout_scores_original_{all,general,constraint}
    rollout_scores_constraint_{all,general,constraint}
    rollout_scores_original_with_original_rubric
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _score_aggregation import aggregate_per_rollout  # noqa: E402


def _augment(rec: dict) -> None:
    weights = rec.get("weights")
    sources = rec.get("criterion_sources") or []

    if "scores_original" in rec and weights is not None:
        agg = aggregate_per_rollout(rec["scores_original"], weights, sources)
        rec["rollout_scores_original_all"] = agg["all"]
        if "general" in agg:
            rec["rollout_scores_original_general"] = agg["general"]
        if "constraint" in agg:
            rec["rollout_scores_original_constraint"] = agg["constraint"]

    if "scores_constraint" in rec and weights is not None:
        agg = aggregate_per_rollout(rec["scores_constraint"], weights, sources)
        rec["rollout_scores_constraint_all"] = agg["all"]
        if "general" in agg:
            rec["rollout_scores_constraint_general"] = agg["general"]
        if "constraint" in agg:
            rec["rollout_scores_constraint_constraint"] = agg["constraint"]

    if "scores_original_with_original_rubric" in rec and "original_general_weights" in rec:
        ow = rec["original_general_weights"]
        agg = aggregate_per_rollout(
            rec["scores_original_with_original_rubric"],
            ow,
            ["general"] * len(ow),
        )
        rec["rollout_scores_original_with_original_rubric"] = agg["all"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Default: overwrite --eval-json in place.")
    args = ap.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {args.eval_json}")

    for rec in records:
        _augment(rec)

    out_path = args.out or args.eval_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Saved -> {out_path}")

    # Quick sanity print: show added fields on first record
    r0 = records[0]
    added_fields = [k for k in r0 if k.startswith("rollout_scores_")]
    print(f"  per-record per-rollout aggregate fields: {added_fields}")


if __name__ == "__main__":
    main()
