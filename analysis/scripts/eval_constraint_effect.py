"""eval_constraint_effect.py

Isolate the pure constraint-addition effect on general rubric score distribution,
free of training dynamics confound.

Procedure
---------
1. Load `kimyuji/generate_constraint_v1` (epoch 0, constraint_added=True),
   subsample ``n_samples`` prompts. Reuses stored general + constraint rubric.
2. Base policy (Qwen3-1.7B) generates ``n_rollouts`` responses for:
   - the original prompt
   - the constraint-added prompt
3. Judge (Qwen3-8B) scores each (rollout, criterion) pair independently
   using the judge template (single-criterion judge).
4. Save per-(condition, rollout, criterion) scores for downstream analysis.

Analysis (post-hoc) compares general-rubric score distributions between the
two conditions: mean/std deltas per prompt and in aggregate.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Reuse the data_generation pipeline modules (LLM build, rollout/judge, IO).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_generation"))

from datasets import load_dataset

from io_utils import load_config, resolve_path, save_records  # noqa: E402
from pipeline_steps import generate_rollouts, score_rollouts  # noqa: E402
from vllm_dp import build_llm  # noqa: E402

# Local helper (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _score_aggregation import aggregate_per_rollout  # noqa: E402


def _split_rubric(record: dict) -> tuple[int, int]:
    """Return (n_general, n_constraint) for an epoch-0 constraint_v1 record."""
    reqs = record["requirements"]
    all_c = record.get("all_constraint_requirements") or record.get("constraint_requirements") or []
    n_c = len(all_c)
    n_g = len(reqs) - n_c
    if n_g < 0:
        raise ValueError(f"Negative n_general: len(reqs)={len(reqs)} n_c={n_c}")
    return n_g, n_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/scripts/configs/eval_constraint_effect.yaml"),
    )
    args = parser.parse_args()

    root = Path.cwd()
    cfg = load_config(resolve_path(str(args.config), root))
    parse_patterns = cfg.get("parse") or {}

    judge_template = resolve_path(cfg["judge_prompt"], root).read_text(encoding="utf-8")

    policy_cfg = cfg["policy_model"]
    judge_cfg = cfg["judge_model"]
    dp_size = cfg.get("data_parallel_size", "auto")
    n_rollouts = int(cfg.get("n_rollouts", 8))
    n_judge_runs = int(cfg.get("n_judge_runs", 1))
    n_samples = int(cfg.get("n_samples", 300))
    seed = int(cfg.get("seed", 42))

    ds_cfg = cfg["dataset"]

    # -------------------------------------------------------------------
    # Load + filter + subsample
    # -------------------------------------------------------------------
    print(f"Loading dataset: {ds_cfg['name']} [{ds_cfg.get('split', 'train')}]")
    load_kwargs = {"split": ds_cfg.get("split", "train")}
    if ds_cfg.get("data_files"):
        load_kwargs["data_files"] = ds_cfg["data_files"]
    full_ds = load_dataset(ds_cfg["name"], **load_kwargs)
    filtered = full_ds.filter(lambda r: r["epoch"] == 0 and r["constraint_added"])
    print(f"  epoch==0 & c_added=True: {len(filtered)} records")

    k = min(n_samples, len(filtered))
    random.seed(seed)
    indices = random.sample(range(len(filtered)), k)
    subset = filtered.select(indices)
    print(f"  Subsampled {k} records (seed={seed})")

    # Build in-memory records with rubric split
    records: list[dict] = []
    for r in subset:
        n_g, n_c = _split_rubric(r)
        reqs = r["requirements"]
        weights = r["weights"]
        records.append(
            {
                "original_prompt": r["original_prompt"],
                "constraint_prompt": r["prompt"],
                "constraint": r.get("constraint", ""),
                "requirements": reqs,
                "weights": weights,
                "n_general": n_g,
                "n_constraint": n_c,
                "general_requirements": reqs[:n_g],
                "general_weights": weights[:n_g],
                "constraint_requirements": reqs[n_g:],
                "constraint_weights": weights[n_g:],
                "criterion_sources": ["general"] * n_g + ["constraint"] * n_c,
            }
        )

    total_criteria = sum(len(r["requirements"]) for r in records)
    print(
        f"  Total criteria: {total_criteria}  "
        f"(general={sum(r['n_general'] for r in records)}, "
        f"constraint={sum(r['n_constraint'] for r in records)})"
    )

    # -------------------------------------------------------------------
    # Policy rollouts: original & constraint-added prompts
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Policy] Load {policy_cfg['name']}\n{'=' * 60}")
    policy = build_llm(policy_cfg["name"], policy_cfg["gpu_memory_utilization"], dp_size)
    policy_tok = policy.get_tokenizer()

    print(f"\n[Rollouts: ORIGINAL] {n_rollouts} per prompt")
    rollouts_orig = generate_rollouts(
        [{"prompt": r["original_prompt"]} for r in records],
        policy,
        policy_tok,
        policy_cfg,
        n_rollouts,
        parse_patterns,
        min_valid=1,
    )

    print(f"\n[Rollouts: CONSTRAINT-ADDED] {n_rollouts} per prompt")
    rollouts_constr = generate_rollouts(
        [{"prompt": r["constraint_prompt"]} for r in records],
        policy,
        policy_tok,
        policy_cfg,
        n_rollouts,
        parse_patterns,
        min_valid=1,
    )

    n_roll_orig = [len(r) for r in rollouts_orig]
    n_roll_constr = [len(r) for r in rollouts_constr]
    print(
        f"\n  Rollouts/prompt (original):   min={min(n_roll_orig)}, "
        f"max={max(n_roll_orig)}, mean={sum(n_roll_orig) / len(n_roll_orig):.2f}"
    )
    print(
        f"  Rollouts/prompt (constraint): min={min(n_roll_constr)}, "
        f"max={max(n_roll_constr)}, mean={sum(n_roll_constr) / len(n_roll_constr):.2f}"
    )

    policy.shutdown()

    # -------------------------------------------------------------------
    # Judge: per-(rollout, criterion) independently
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Judge] Load {judge_cfg['name']}\n{'=' * 60}")
    judge = build_llm(judge_cfg["name"], judge_cfg["gpu_memory_utilization"], dp_size)
    judge_tok = judge.get_tokenizer()

    print("\n[Scoring: ORIGINAL] judge per criterion")
    orig_samples = [
        {
            "prompt": r["original_prompt"],
            "requirements": r["requirements"],
            "weights": r["weights"],
        }
        for r in records
    ]
    scores_orig, justs_orig = score_rollouts(
        orig_samples,
        rollouts_orig,
        judge,
        judge_tok,
        judge_cfg,
        judge_template,
        parse_patterns,
        n_judge_runs,
    )

    print("\n[Scoring: CONSTRAINT-ADDED] judge per criterion")
    constr_samples = [
        {
            "prompt": r["constraint_prompt"],
            "requirements": r["requirements"],
            "weights": r["weights"],
        }
        for r in records
    ]
    scores_constr, justs_constr = score_rollouts(
        constr_samples,
        rollouts_constr,
        judge,
        judge_tok,
        judge_cfg,
        judge_template,
        parse_patterns,
        n_judge_runs,
    )

    judge.shutdown()

    # -------------------------------------------------------------------
    # Attach scores to records and save
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Save] Build records\n{'=' * 60}")
    for si, rec in enumerate(records):
        n_k = len(rec["requirements"])
        n_r_o = len(rollouts_orig[si])
        n_r_c = len(rollouts_constr[si])
        rec["rollouts_original"] = rollouts_orig[si]
        rec["rollouts_constraint"] = rollouts_constr[si]
        # shape: [n_rollouts][n_criteria][n_judge_runs]
        rec["scores_original"] = [
            [scores_orig.get((si, rr, k), []) for k in range(n_k)] for rr in range(n_r_o)
        ]
        rec["scores_constraint"] = [
            [scores_constr.get((si, rr, k), []) for k in range(n_k)] for rr in range(n_r_c)
        ]
        rec["justifications_original"] = [
            [justs_orig.get((si, rr, k), []) for k in range(n_k)] for rr in range(n_r_o)
        ]
        rec["justifications_constraint"] = [
            [justs_constr.get((si, rr, k), []) for k in range(n_k)] for rr in range(n_r_c)
        ]

        # Per-rollout weighted-mean aggregates (saved alongside raw scores so
        # analysis scripts can read scalars directly without recomputing).
        agg_o = aggregate_per_rollout(rec["scores_original"], rec["weights"], rec["criterion_sources"])
        agg_c = aggregate_per_rollout(rec["scores_constraint"], rec["weights"], rec["criterion_sources"])
        rec["rollout_scores_original_all"]     = agg_o["all"]
        rec["rollout_scores_constraint_all"]   = agg_c["all"]
        if "general" in agg_o:
            rec["rollout_scores_original_general"]   = agg_o["general"]
            rec["rollout_scores_constraint_general"] = agg_c["general"]
        if "constraint" in agg_o:
            rec["rollout_scores_original_constraint"]   = agg_o["constraint"]
            rec["rollout_scores_constraint_constraint"] = agg_c["constraint"]

    save_records(records, cfg["output"], root)
    print("Done.")


if __name__ == "__main__":
    main()
