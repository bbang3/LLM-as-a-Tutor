# analysis

Post-hoc analysis of policy behavior on the constraint-evolving training runs.

## Layout

- `scripts/` — runnable Python scripts (details below)
- `scripts/configs/` — per-script YAML configs (input paths, model/rollout settings)
- `findings/` — hand-written insight writeups (markdown); not for script outputs
- `outputs/` — script outputs (CSV / JSON / PNG). All eval JSONs land here.

Two largely independent pipelines live in here:

1. **Epoch-difficulty pipeline** — measures how the base policy behaves
   across the iterative training epochs (rubric grows with constraints).
2. **Constraint-effect pipeline** — at a single base-policy snapshot,
   compare rollouts on bare `original_prompt` vs. constraint-added prompt
   under two methodologies: append (`constraint_generation`) and rewrite
   (`rewrite_variant`).

Plus a third, smaller **training-trajectory pipeline** (`merge_rollouts`,
`rubric_score_trajectory`) that consumes the trainer's per-step rollout
JSONLs to plot reward over time.

---

## Pipeline 1 — Epoch difficulty (run in order)

- **`sample_epoch_prompts.py`** — sample N `original_prompt`s that appear in
  all epochs and had a constraint added at some point. Emits a JSONL
  bundling each prompt's per-epoch input + rubric.
- **`eval_epoch_difficulty.py`** — for each sampled (prompt, epoch), base
  policy generates rollouts on the epoch-specific `input` and judge scores
  every criterion. Writes merged JSONL (epoch entries with
  rollouts/scores/justifications).
- **`eval_epoch_difficulty_baseline.py`** — augments the results file with
  a no-constraint baseline: rollouts on `original_prompt` scored against
  `general_requirements` only.

Analysis (consume the `eval_epoch_difficulty*` JSONL):

- **`rubric_score_distribution.py`** — per-epoch aggregate mean/std/median
  of the per-prompt score for a configurable subset of criterion categories
  (`include_sources` in the config; e.g. `[general]`, or
  `[general, constraint, adaptive]` for the total). Also per-prompt
  monotonicity and pairwise transition deltas. Optional markdown/figure outputs.
- **`constraint_counts.py`** — per-epoch rubric-growth stats: newly added
  constraints, total constraints, cumulative versions, and the dataset's
  `constraint_added` flag counts.

---

## Pipeline 2 — Constraint-effect (eval-style, single snapshot)

For a given base policy, compare rollouts on `original_prompt` vs. the
constraint-added/rewritten variant under the matching rubric. Two
methodologies, two parallel sub-pipelines.

### 2a. Build the dataset

- **`build_constraint_effect_dataset.py`** (append / v1 method) —
  Convert a training checkpoint's `dataset_snapshot.parquet` (filtered to
  `constraint_added=True`) into the parquet schema consumed by
  `eval_constraint_effect.py`. Used when the training run already produced
  v1-style appended-constraint data (e.g. `llm_tutor*`).
- **`build_rewrite_constraint_dataset.py`** (rewrite / v2 method) — Reuses
  base-policy responses from the training run's
  `data_generator/constraint_judgment.jsonl`, runs `rewrite_variant.txt`
  with the 8B generator, then regenerates the general rubric on each
  rewritten prompt with `base_rubric_generation.txt` (rewrite changes the task,
  so the original rubric no longer applies). Outputs an
  eval-script-compatible parquet.

### 2b. Run the eval

- **`eval_constraint_effect.py`** — for every record in the dataset,
  base-policy generates `n_rollouts` on both `original_prompt` and
  `constraint_prompt`; judge scores every (rollout, criterion) pair with
  `judge.txt`. Saves a JSON with raw scores plus per-rollout
  weighted-mean aggregates (see § *Per-rollout aggregates* below).

### 2c. Rescore (rewrite path only)

The rewrite changes the task, so the original-prompt rollouts must be
scored against the **original** general rubric (kept in the parquet under
`original_general_requirements`), not the rewrite's regenerated rubric:

- **`rescore_original_with_original_rubric.py`** — re-runs the 8B judge
  over `rollouts_original` × `original_general_requirements` and adds
  `scores_original_with_original_rubric` (and matching aggregates) to the
  eval JSON.

### 2d. Analyze

- **`paired_delta_constraint_effect.py`** — paired Δ (treated − original)
  on per-prompt mean and within-prompt std, plus pooled per-rollout
  distribution. Auto-detects v1 (shared rubric, filter to general criteria)
  vs. v2 (per-condition rubric via the rescore output) by record contents;
  raises if a v2-style file hasn't been rescored.

### 2e. Per-rollout aggregates

`eval_constraint_effect.py` and `rescore_original_with_original_rubric.py`
both store per-rollout weighted-mean scalars alongside the raw 3-D score
tensor, so analysis scripts read scalars directly without recomputing.

Fields stored per record (subject to existence of source criteria):
```
rollout_scores_original_all          rollout_scores_constraint_all
rollout_scores_original_general      rollout_scores_constraint_general
rollout_scores_original_constraint   rollout_scores_constraint_constraint
rollout_scores_original_with_original_rubric    # rescore output only
```
Each is a `list[float | None]` of length `n_rollouts` (None when every
judge call returned parse_fail).

- **`augment_per_rollout_aggregates.py`** — retroactively adds these
  aggregate fields to an existing eval JSON. Idempotent. Run once on legacy
  outputs that pre-date the auto-save change.

### Typical end-to-end flow (rewrite path)

```bash
# 1. build dataset from training-run snapshot + judgment.jsonl
python analysis/scripts/build_rewrite_constraint_dataset.py \
    --judgment-jsonl checkpoints/.../data_generator/constraint_judgment.jsonl \
    --snapshot       checkpoints/.../global_step_124/dataset_snapshot.parquet \
    --out            datasets/eval_constraint_effect_<run>_rewrite_v2_ep1/train.parquet \
    --max-samples    800

# 2. eval (rollouts on original + rewrite, judge with regen rubric)
python analysis/scripts/eval_constraint_effect.py \
    --config analysis/scripts/configs/eval_constraint_effect_8b_rewrite.yaml

# 3. rescore original-condition rollouts with the original rubric
python analysis/scripts/rescore_original_with_original_rubric.py \
    --eval-json       analysis/outputs/<run>_rewrite_v2_ep1.json \
    --dataset-parquet datasets/eval_constraint_effect_<run>_rewrite_v2_ep1/train.parquet

# 4. paired Δ
python analysis/scripts/paired_delta_constraint_effect.py \
    --eval-json analysis/outputs/<run>_rewrite_v2_ep1_rescored.json \
    --out-csv   analysis/outputs/paired_delta_<run>_v2.csv
```

### Append (v1) path

```bash
# 1. build dataset directly from snapshot's constraint_added=True rows
python analysis/scripts/build_constraint_effect_dataset.py \
    --snapshot checkpoints/.../global_step_124/dataset_snapshot.parquet \
    --out      datasets/eval_constraint_effect_<run>_ep1/train.parquet

# 2. eval
python analysis/scripts/eval_constraint_effect.py \
    --config analysis/scripts/configs/eval_constraint_effect_8b.yaml

# 3. paired Δ (no rescore — both conditions share the same rubric)
python analysis/scripts/paired_delta_constraint_effect.py \
    --eval-json analysis/outputs/<run>_ep1.json \
    --out-csv   analysis/outputs/paired_delta_<run>_v1.csv
```

---

## Pipeline 3 — Training-trajectory plots

Reads the trainer's `rollouts/{step}.jsonl` files (one per training step)
from a checkpoint dir.

- **`merge_rollouts.py`** — merges the per-step JSONLs into one slim
  `rollouts_merged.jsonl` (one row per rollout) and attaches per-criterion
  source labels by joining with each epoch's `dataset_snapshot.parquet`.
- **`rubric_score_trajectory.py`** — per-step trajectory of rubric scores
  (general / constraint / all bundles), with EMA smoothing reset at
  detected epoch boundaries. Also a per-prompt epoch-comparison figure.

---

## Shared

- **`_common.py`** — schema docstring, JSONL loader + validator,
  criterion-source filtering, per-rollout (un)weighted mean helpers,
  `safe_mean / stdev / median`. `safe_stdev` uses `statistics.stdev`
  (ddof=1, Bessel's correction) to match GRPO advantage normalization.
- **`_score_aggregation.py`** — `aggregate_per_rollout()` used by the
  constraint-effect pipeline (eval + rescore + augment) to compute
  per-rollout weighted-mean scalars from the raw 3-D score tensor, split
  by `criterion_sources` bucket.
