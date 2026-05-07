"""sample_epoch_prompts.py

Sample original_prompts that (a) appear in all training epochs and (b) had a
constraint added in at least one epoch. For each sampled original_prompt, emit
a single record bundling its per-epoch prompt + rubric. Downstream inference
loops over the per-epoch entries to test whether prompt difficulty grows
monotonically across epochs.

Output (JSONL, one record per original_prompt):
    {
      "prompt_id": int,
      "original_prompt": str,
      "epochs": [
        {
          "epoch": int,
          "input": str,
          "constraint_added": bool,
          "requirements": [...], "weights": [...],
          "general_requirements": [...], "general_weights": [...],
          "constraint_requirements": [...], "constraint_weights": [...],
          "adaptive_requirements": [...], "adaptive_weights": [...],
          "criterion_sources": [...],
          "all_constraint_requirements": [...], "all_constraint_weights": [...]
        },
        ...  # one per epoch, sorted ascending
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

EPOCH_FIELDS = (
    "epoch",
    "input",
    "constraint_added",
    "requirements",
    "weights",
    "general_requirements",
    "general_weights",
    "constraint_requirements",
    "constraint_weights",
    "adaptive_requirements",
    "adaptive_weights",
    "criterion_sources",
    "all_constraint_requirements",
    "all_constraint_weights",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/shared/sangminhwang/pllm_logs/"
            "rubric_policy_v1.4_constraint_4k_all_epochs_merged_slim.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/shared/sangminhwang/pllm_logs/"
            "eval_epoch_difficulty_sample100.jsonl"
        ),
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    # group rows by (original_prompt, epoch); within a group all rollouts share
    # the same prompt/rubric so we keep the first row only.
    by_op_ep: dict[tuple[str, int], dict] = {}
    n_lines = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            d = json.loads(line)
            key = (d["original_prompt"], d["epoch"])
            if key not in by_op_ep:
                by_op_ep[key] = d
    print(f"  read {n_lines} rows -> {len(by_op_ep)} unique (op, epoch) groups")

    op_to_epochs: dict[str, dict[int, dict]] = defaultdict(dict)
    for (op, ep), row in by_op_ep.items():
        op_to_epochs[op][ep] = row
    print(f"  unique original_prompts: {len(op_to_epochs)}")

    all_epoch_ids = sorted({ep for eps in op_to_epochs.values() for ep in eps})
    required_epochs = set(all_epoch_ids)
    print(f"  epochs present in dataset: {sorted(required_epochs)}")

    qualifying: list[str] = []
    for op, eps in op_to_epochs.items():
        if set(eps.keys()) != required_epochs:
            continue
        if not any(eps[e]["constraint_added"] for e in eps):
            continue
        qualifying.append(op)
    print(
        f"  qualifying (all epochs present AND constraint_added in >=1 epoch): "
        f"{len(qualifying)}"
    )

    if args.n_samples > len(qualifying):
        raise ValueError(
            f"n_samples={args.n_samples} > qualifying pool {len(qualifying)}"
        )

    qualifying.sort()  # deterministic order before sampling
    rng = random.Random(args.seed)
    sampled = rng.sample(qualifying, args.n_samples)
    print(f"  sampled {len(sampled)} (seed={args.seed})")

    records: list[dict] = []
    for pid, op in enumerate(sampled):
        eps = op_to_epochs[op]
        epoch_entries = []
        for e in sorted(eps):
            row = eps[e]
            epoch_entries.append({k: row.get(k) for k in EPOCH_FIELDS})
        records.append(
            {
                "prompt_id": pid,
                "original_prompt": op,
                "epochs": epoch_entries,
            }
        )

    n_ca_per_epoch = {
        e: sum(1 for r in records for ee in r["epochs"] if ee["epoch"] == e and ee["constraint_added"])
        for e in sorted(required_epochs)
    }
    print(f"  constraint_added=True counts per epoch: {n_ca_per_epoch}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
