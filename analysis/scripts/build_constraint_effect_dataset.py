"""build_constraint_effect_dataset.py

Build a dataset for ``eval_constraint_effect.py`` from a training-checkpoint
snapshot.

Why
---
The original ``eval_constraint_effect.py`` consumes
``kimyuji/generate_constraint_v1`` — constraints that an 8B generator emitted
on a *1.7B* base policy's outputs. To evaluate the constraint-addition effect
on a different base (e.g. Qwen3-8B), we want the constraints generated for
*that* base policy's outputs.

The first ever data-generation step of a training run runs on the untrained
base policy (no gradient updates yet). The dataset state immediately after
that step lives in the epoch-1 ``dataset_snapshot.parquet``. Filtering to
``constraint_added=True`` gives the analog of the kimyuji v1 dataset for the
training run's base policy.

Output schema (matches the kimyuji v1 fields ``eval_constraint_effect.py``
reads):
  - ``epoch`` (always 0; needed by the script's filter)
  - ``constraint_added`` (always True)
  - ``original_prompt``
  - ``prompt`` (= original_prompt + appended constraint)
  - ``constraint`` (the appended text)
  - ``requirements`` (general + constraint, in that order)
  - ``weights``
  - ``constraint_requirements`` (constraint-only, used by ``_split_rubric``)
  - ``constraint_weights``

Usage
-----
  python analysis/scripts/build_constraint_effect_dataset.py \
      --snapshot checkpoints/.../global_step_124/dataset_snapshot.parquet \
      --out datasets/eval_constraint_effect_v14_8b_ep1/train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_list(v):
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to dataset_snapshot.parquet (typically global_step_<N>/dataset_snapshot.parquet "
        "where <N> is the first epoch's last step — that snapshot reflects the dataset state "
        "after the first generation step on the untrained base policy).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output parquet path. Parent dirs are created.",
    )
    args = parser.parse_args()

    snap = pd.read_parquet(args.snapshot)
    print(f"Loaded {len(snap)} rows from {args.snapshot}")

    ca = snap[snap["constraint_added"].astype(bool)].copy()
    print(f"  constraint_added=True: {len(ca)}")

    records = []
    n_skip_no_prefix = 0
    n_skip_empty_constr = 0
    for _, r in ca.iterrows():
        prompt = r["prompt"]
        op = r["original_prompt"]
        # Extract appended constraint text. The training pipeline appends
        # "\nAdditionally, ..." (sometimes after rstrip on the original).
        op_strip = op.rstrip()
        if prompt.startswith(op_strip):
            constraint_text = prompt[len(op_strip):].lstrip("\n").strip()
        else:
            n_skip_no_prefix += 1
            continue
        if not constraint_text:
            n_skip_empty_constr += 1
            continue

        reqs = _to_list(r["requirements"])
        weights = [int(w) for w in _to_list(r["weights"])]
        cr = _to_list(r.get("constraint_requirements"))
        cw_raw = _to_list(r.get("constraint_weights"))
        cw = [int(w) for w in cw_raw]

        records.append({
            "epoch": 0,
            "constraint_added": True,
            "prompt": prompt,
            "original_prompt": op,
            "constraint": constraint_text,
            "requirements": reqs,
            "weights": weights,
            "constraint_requirements": cr,
            "constraint_weights": cw,
        })

    print(f"  built {len(records)} records (skipped: prefix-mismatch={n_skip_no_prefix}, empty-constraint={n_skip_empty_constr})")

    df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved -> {args.out}  ({len(df)} rows)")

    # Quick sanity stats
    n_g = df["requirements"].apply(len) - df["constraint_requirements"].apply(len)
    n_c = df["constraint_requirements"].apply(len)
    print(f"  per-prompt n_general:    mean={n_g.mean():.2f}, min={n_g.min()}, max={n_g.max()}")
    print(f"  per-prompt n_constraint: mean={n_c.mean():.2f}, min={n_c.min()}, max={n_c.max()}")


if __name__ == "__main__":
    main()
