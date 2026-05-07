"""eval_epoch_difficulty.py

Test whether the per-epoch prompts in a constraint-evolving training run
get monotonically harder for a fixed base policy.

Input
-----
A JSONL produced by ``sample_epoch_prompts.py``: one record per
``original_prompt``, each with a list of epoch entries (``input``,
``requirements``, ``weights``, ...).

Procedure
---------
1. Build a single batch of samples covering, per record:
   - one ``baseline`` sample using ``original_prompt`` (no constraint)
   - one sample per epoch using the epoch-specific ``input``
2. Base policy generates ``n_rollouts`` responses for every sample.
3. Judge scores every (rollout, criterion) pair using the judge template.
   - Baseline rollouts are scored against ``general_requirements`` only (the
     general rubric is fixed across epochs, so this gives a true
     zero-constraint baseline directly comparable to per-epoch general
     scores).
   - Epoch rollouts are scored against the epoch's full ``requirements``.
4. Save results, re-grouped by ``prompt_id``: a top-level ``baseline`` field
   and per-epoch entries with rollouts/scores/justifications attached.

Downstream analysis: compare per-epoch mean scores against the baseline
(``record["baseline"]``) and across epochs (``record["epochs"][i]``). Restrict
to ``criterion_sources == "general"`` for the cleanest cross-epoch signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the data_generation pipeline modules (LLM build, rollout/judge, IO).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_generation"))

from io_utils import load_config, resolve_path, save_records  # noqa: E402
from pipeline_steps import generate_rollouts, score_rollouts  # noqa: E402
from vllm_dp import build_llm  # noqa: E402


def _load_sample_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/scripts/configs/eval_epoch_difficulty.yaml"),
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

    input_path = resolve_path(cfg["input_path"], root)
    print(f"Loading samples: {input_path}")
    records = _load_sample_file(input_path)
    print(f"  records: {len(records)}")

    # -------------------------------------------------------------------
    # Flatten samples: 1 baseline per record + 1 per (record, epoch)
    # -------------------------------------------------------------------
    flat: list[dict] = []
    # flat_index[fi] is either ("baseline", record_i) or ("epoch", record_i, epoch_i)
    flat_index: list[tuple] = []

    for ri, rec in enumerate(records):
        # baseline: original_prompt scored against general criteria (fixed across epochs)
        ep0 = rec["epochs"][0]
        gen_reqs = ep0.get("general_requirements") or []
        gen_weights = ep0.get("general_weights") or []
        if len(gen_reqs) != len(gen_weights):
            raise ValueError(
                f"prompt_id={rec.get('prompt_id')}: general_requirements/general_weights length mismatch"
            )
        flat.append(
            {
                "prompt": rec["original_prompt"],
                "requirements": gen_reqs,
                "weights": gen_weights,
            }
        )
        flat_index.append(("baseline", ri))

        for ei, ep in enumerate(rec["epochs"]):
            flat.append(
                {
                    "prompt": ep["input"],
                    "requirements": ep["requirements"],
                    "weights": ep["weights"],
                }
            )
            flat_index.append(("epoch", ri, ei))

    n_baseline = sum(1 for x in flat_index if x[0] == "baseline")
    n_epoch = sum(1 for x in flat_index if x[0] == "epoch")
    n_epochs_per_record = [len(r["epochs"]) for r in records]
    print(
        f"  flattened: {len(flat)} samples = {n_baseline} baseline + {n_epoch} (prompt, epoch) "
        f"(epochs/record min={min(n_epochs_per_record)}, max={max(n_epochs_per_record)})"
    )
    total_criteria = sum(len(s["requirements"]) for s in flat)
    print(f"  total criteria across all samples: {total_criteria}")

    # -------------------------------------------------------------------
    # Policy rollouts
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Policy] Load {policy_cfg['name']}\n{'=' * 60}")
    policy = build_llm(policy_cfg["name"], policy_cfg["gpu_memory_utilization"], dp_size)
    policy_tok = policy.get_tokenizer()

    print(f"\n[Rollouts] {n_rollouts} per sample (baseline + per-epoch)")
    rollouts = generate_rollouts(
        flat,
        policy,
        policy_tok,
        policy_cfg,
        n_rollouts,
        parse_patterns,
        min_valid=1,
    )

    n_roll = [len(r) for r in rollouts]
    print(
        f"\n  Rollouts/sample: min={min(n_roll)}, max={max(n_roll)}, "
        f"mean={sum(n_roll) / len(n_roll):.2f}"
    )

    policy.shutdown()

    # -------------------------------------------------------------------
    # Judge: per-criterion independently
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Judge] Load {judge_cfg['name']}\n{'=' * 60}")
    judge = build_llm(judge_cfg["name"], judge_cfg["gpu_memory_utilization"], dp_size)
    judge_tok = judge.get_tokenizer()

    print("\n[Scoring] judge per criterion")
    scores, justs = score_rollouts(
        flat,
        rollouts,
        judge,
        judge_tok,
        judge_cfg,
        judge_template,
        parse_patterns,
        n_judge_runs,
    )

    judge.shutdown()

    # -------------------------------------------------------------------
    # Re-group results back into per-record / per-epoch structure
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}\n[Save] Build records\n{'=' * 60}")
    for fi, key in enumerate(flat_index):
        sample = flat[fi]
        n_k = len(sample["requirements"])
        n_r = len(rollouts[fi])
        rollout_list = rollouts[fi]
        # shape: [n_rollouts][n_criteria][n_judge_runs]
        scores_grid = [
            [scores.get((fi, rr, k), []) for k in range(n_k)] for rr in range(n_r)
        ]
        justs_grid = [
            [justs.get((fi, rr, k), []) for k in range(n_k)] for rr in range(n_r)
        ]
        if key[0] == "baseline":
            ri = key[1]
            records[ri]["baseline"] = {
                "prompt": sample["prompt"],
                "requirements": sample["requirements"],
                "weights": sample["weights"],
                "criterion_sources": ["general"] * n_k,
                "rollouts": rollout_list,
                "scores": scores_grid,
                "justifications": justs_grid,
            }
        else:  # ("epoch", ri, ei)
            _, ri, ei = key
            ep = records[ri]["epochs"][ei]
            ep["rollouts"] = rollout_list
            ep["scores"] = scores_grid
            ep["justifications"] = justs_grid

    # Shift stored epoch ids to 1-indexed on the way out so downstream
    # labels match training-epoch terminology (e1 = first training epoch).
    # The input file (from ``sample_epoch_prompts.py``) remains 0-indexed;
    # only this script's output JSONL is 1-indexed.
    for rec in records:
        for ep in rec["epochs"]:
            ep["epoch"] = int(ep["epoch"]) + 1

    save_records(records, cfg["output"], root)
    print("Done.")


if __name__ == "__main__":
    main()
