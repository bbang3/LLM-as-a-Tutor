"""constraint_addition_rate.py

Per-run constraint-addition rate from ``data_generator/epoch_state/epoch_*.pt``.

Each ``epoch_<N>.pt`` is the FINAL applied dataset state at the end of epoch N
(retries collapsed). For every prompt row it stores ``requirements_constraint``;
non-empty ⇔ a constraint was added to that row for the next epoch's training.

For each (run, epoch) reports:
  - total rows
  - rows with constraint_added=True (= ``len(requirements_constraint) > 0``)
  - rate (%)
  - mean / min / max n_constraint per added row

CLI usage
---------
  python analysis/scripts/constraint_addition_rate.py \
      --runs llm_tutor_policy_0.6b \
             llm_tutor_policy_4b \
             llm_tutor_policy_8b

  # Or globbed from default checkpoints base:
  python analysis/scripts/constraint_addition_rate.py --glob 'llm_tutor_policy_*'
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

DEFAULT_CKPT_BASE = Path("checkpoints/prompting-llm-verl")


def discover_epoch_files(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch_int, path), ...] sorted by epoch for a run."""
    state_dir = run_dir / "data_generator" / "epoch_state"
    if not state_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in state_dir.glob("epoch_*.pt"):
        m = re.match(r"epoch_(\d+)\.pt$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def epoch_state_stats(path: Path) -> dict:
    """Compute constraint-addition stats from a single ``epoch_*.pt``."""
    d = torch.load(path, weights_only=False)
    ro = d["state"]["row_overrides"]
    n_total = len(ro)
    n_constraints = [
        len(v.get("requirements_constraint") or []) for v in ro.values()
    ]
    added = [n for n in n_constraints if n > 0]
    return {
        "n_total": n_total,
        "n_added": len(added),
        "rate_pct": 100.0 * len(added) / n_total if n_total else 0.0,
        "nc_mean": (sum(added) / len(added)) if added else 0.0,
        "nc_min": min(added) if added else 0,
        "nc_max": max(added) if added else 0,
    }


def collect(runs: list[Path]) -> list[dict]:
    """Walk runs × epochs and collect rows for the report."""
    rows: list[dict] = []
    for run_dir in runs:
        files = discover_epoch_files(run_dir)
        if not files:
            print(f"[skip] no epoch_state files under {run_dir}", file=sys.stderr)
            continue
        for epoch, path in files:
            s = epoch_state_stats(path)
            rows.append({"run": run_dir.name, "epoch": epoch, **s})
    return rows


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        return "*(no rows)*\n"
    header = (
        "| run | epoch | rows | added | rate% | nc/row mean | nc/row min | nc/row max |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    body = []
    for r in rows:
        body.append(
            f"| {r['run']} | {r['epoch']} | {r['n_total']} | {r['n_added']} | "
            f"{r['rate_pct']:.2f} | {r['nc_mean']:.2f} | {r['nc_min']} | {r['nc_max']} |"
        )
    return header + "\n" + "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Run names (relative to --checkpoints-base) or absolute run dirs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern (relative to --checkpoints-base) to expand into runs, "
             "e.g. 'llm_tutor_policy_*'.",
    )
    parser.add_argument(
        "--checkpoints-base",
        type=Path,
        default=DEFAULT_CKPT_BASE,
        help=f"Base dir for run name resolution (default: {DEFAULT_CKPT_BASE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write markdown output (in addition to stdout).",
    )
    args = parser.parse_args()

    base = args.checkpoints_base.resolve()

    run_dirs: list[Path] = []
    for r in args.runs:
        p = Path(r)
        run_dirs.append(p if p.is_absolute() else base / p)
    if args.glob:
        run_dirs.extend(sorted(base.glob(args.glob)))
    # de-duplicate while preserving order
    seen: set[Path] = set()
    unique_runs: list[Path] = []
    for p in run_dirs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_runs.append(p)

    if not unique_runs:
        parser.error("No runs specified. Pass --runs and/or --glob.")

    rows = collect(unique_runs)
    md = render_markdown(rows)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Saved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
