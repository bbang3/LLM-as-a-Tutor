"""constraint_counts.py

Per-epoch counts describing how the rubric grows across epochs:
  - newly_added_constraints  : per-prompt (n_constraint at e) - (n_constraint at e-1);
                               for e=epochs[0], baseline = 0
  - total_constraint_requirements : per-prompt n_constraint at this epoch
                                    (= cumulative # constraint criteria so far)
  - cumulative_newly_added   : per-prompt running sum of newly_added across epochs
                               (equals total_constraint_requirements when constraints
                               are append-only, which is the assumed pipeline behavior)
  - cumulative_total         : per-prompt running sum of total_constraint_requirements
                               (= "constraint-criterion epoch-exposure", i.e. how many
                               constraint criteria the model has seen up to this epoch)

For each metric, reports per-epoch mean / std / sum across prompts. Optional
breakdown by ``constraint_added`` flag (the dataset's per-row marker).

Inputs (config YAML)
--------------------
  input_path: JSONL produced by ``eval_epoch_difficulty.py`` (schema in _common.py)
              (Only the rubric metadata is read — scores/rollouts ignored.)

Output is printed to stdout as a markdown table; nothing is written to disk.

CLI usage
---------
  python analysis/scripts/constraint_counts.py \
      --config analysis/scripts/configs/constraint_counts.yaml
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from _common import (
    discover_epochs,
    filter_idx,
    fmt,
    load_config,
    load_records,
    safe_mean,
    safe_stdev,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/scripts/configs/constraint_counts.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = Path(cfg["input_path"]).expanduser()

    print(f"Loading: {input_path}")
    records = load_records(input_path)
    epochs = discover_epochs(records)
    print(f"  records={len(records)}, epochs={epochs}")

    # ---------- per-prompt per-epoch counts ----------
    # added[i][j], total[i][j], cum_added[i][j], cum_total[i][j], flag[i][j]
    n = len(records)
    added: list[list[int | None]] = [[None] * len(epochs) for _ in range(n)]
    total: list[list[int | None]] = [[None] * len(epochs) for _ in range(n)]
    cum_added: list[list[int | None]] = [[None] * len(epochs) for _ in range(n)]
    cum_total: list[list[int | None]] = [[None] * len(epochs) for _ in range(n)]
    flag_added: list[list[bool | None]] = [[None] * len(epochs) for _ in range(n)]

    for i, rec in enumerate(records):
        by_epoch = {ep["epoch"]: ep for ep in rec["epochs"]}
        prev_total = 0
        running_added = 0
        running_total = 0
        for j, e in enumerate(epochs):
            ep = by_epoch.get(e)
            if ep is None:
                continue
            n_con = len(filter_idx(ep, "constraint"))
            n_new = n_con - prev_total
            running_added += max(n_new, 0)
            running_total += n_con
            added[i][j] = n_new
            total[i][j] = n_con
            cum_added[i][j] = running_added
            cum_total[i][j] = running_total
            flag_added[i][j] = bool(ep.get("constraint_added"))
            prev_total = n_con

    # ---------- aggregate per epoch ----------
    epoch_stats: list[dict] = []
    for j, e in enumerate(epochs):
        col_added = [added[i][j] for i in range(n) if added[i][j] is not None]
        col_total = [total[i][j] for i in range(n) if total[i][j] is not None]
        col_cum_a = [cum_added[i][j] for i in range(n) if cum_added[i][j] is not None]
        col_cum_t = [cum_total[i][j] for i in range(n) if cum_total[i][j] is not None]
        col_flag = [flag_added[i][j] for i in range(n) if flag_added[i][j] is not None]

        epoch_stats.append({
            "epoch": e,
            "n_prompts": len(col_total),
            "newly_added_constraints": {
                "mean": safe_mean(col_added),
                "std": safe_stdev(col_added),
                "sum": sum(col_added) if col_added else 0,
                "n_prompts_with_addition": sum(1 for x in col_added if x and x > 0),
            },
            "total_constraint_requirements": {
                "mean": safe_mean(col_total),
                "std": safe_stdev(col_total),
                "sum": sum(col_total) if col_total else 0,
                "n_prompts_with_any": sum(1 for x in col_total if x and x > 0),
            },
            "cumulative_newly_added": {
                "mean": safe_mean(col_cum_a),
                "std": safe_stdev(col_cum_a),
                "sum": sum(col_cum_a) if col_cum_a else 0,
            },
            "cumulative_total": {
                "mean": safe_mean(col_cum_t),
                "std": safe_stdev(col_cum_t),
                "sum": sum(col_cum_t) if col_cum_t else 0,
            },
            "constraint_added_flag_true": sum(1 for x in col_flag if x),
        })

    # ---------- markdown ----------
    lines: list[str] = []
    lines.append("# Constraint counts per epoch\n")
    lines.append(f"Input: `{input_path}`\n")
    lines.append("\n## Per-epoch aggregates (across prompts)\n")
    lines.append(
        "| epoch | n_prompts | newly_added (mean) | newly_added (sum) | "
        "total (mean) | total (sum) | "
        "cum_newly_added (mean) | cum_total (mean) | "
        "flag_ca=True (#prompts) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in epoch_stats:
        lines.append(
            f"| {s['epoch']} | {s['n_prompts']} | "
            f"{fmt(s['newly_added_constraints']['mean'], '.2f')} | "
            f"{s['newly_added_constraints']['sum']} | "
            f"{fmt(s['total_constraint_requirements']['mean'], '.2f')} | "
            f"{s['total_constraint_requirements']['sum']} | "
            f"{fmt(s['cumulative_newly_added']['mean'], '.2f')} | "
            f"{fmt(s['cumulative_total']['mean'], '.2f')} | "
            f"{s['constraint_added_flag_true']} |"
        )

    lines.append("\n### Field definitions\n")
    lines.append("- **newly_added**: `n_constraint(e) - n_constraint(e-1)` per prompt; epoch 0 baseline = 0.")
    lines.append("- **total**: `n_constraint(e)` per prompt (= cumulative criteria, since the pipeline appends).")
    lines.append("- **cum_newly_added**: per-prompt running sum of `newly_added` across epochs ≤ e.")
    lines.append("- **cum_total**: per-prompt running sum of `total` across epochs ≤ e (criterion-epoch exposures).")
    lines.append(
        "- **flag_ca=True**: number of prompts where the dataset's `constraint_added` boolean is True at this epoch.\n"
    )

    md = "\n".join(lines) + "\n"

    print()
    print(md)


if __name__ == "__main__":
    main()
