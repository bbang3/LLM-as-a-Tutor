"""Shared utilities for epoch-difficulty analysis scripts.

Expected input data format (JSONL, one record per ``original_prompt``)
---------------------------------------------------------------------
Two producers emit this schema; scripts only read fields they need, so the
schema is split into a **required core** (enforced by ``load_records``) and
**optional extensions** (populated when available).

Required per record
~~~~~~~~~~~~~~~~~~~
    prompt_id:       int
    original_prompt: str
    epochs:          list[dict] (each with the required epoch fields below)

Required per epoch entry
~~~~~~~~~~~~~~~~~~~~~~~~
    epoch:              int
    requirements:       list[str]         # full rubric (general + constraint + adaptive)
    weights:            list[int]         # one weight per criterion
    criterion_sources:  list[str]         # element in {"general","constraint","adaptive"}

Optional per record (common)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    baseline: {prompt, requirements, weights, criterion_sources, rollouts,
               scores, justifications}
        Present when ``eval_epoch_difficulty.py`` ran with a no-constraint
        baseline (rollouts on bare ``original_prompt`` scored against
        ``general_requirements``).

Optional per epoch entry
~~~~~~~~~~~~~~~~~~~~~~~~
    input:                      str       # epoch-specific prompt (with constraints)
    constraint_added:           bool      # dataset-level flag
    general_requirements:       list[str]
    general_weights:            list[int]
    constraint_requirements:    list[str]
    constraint_weights:         list[int]
    adaptive_requirements:      list[str] | None
    adaptive_weights:           list[int] | None
    rollouts:         list[str]             # base-policy rollouts for ``input``
    scores:           list[list[list[int]]] # shape [n_rollouts][n_criteria][n_judge_runs];
                                            # -1 entries mean judge parse failure
    justifications:   list[list[list[str]]] # same shape as ``scores``

Producers
~~~~~~~~~
    ``eval_epoch_difficulty.py``: fills everything including rollouts/scores
                                  (always has ``input``, ``constraint_added``).
    Snapshot mergers (e.g. ``merge_dataset_snapshots.py``): fill the rubric
                                  state but may omit rollouts/scores.

Scripts that need rollouts/scores must defensively check for them (e.g.
``ep.get("rollouts")``) rather than assuming their presence.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from pathlib import Path

import yaml

REQUIRED_RECORD_FIELDS = ("prompt_id", "original_prompt", "epochs")
# Structural metadata required on every epoch entry. ``scores`` / ``rollouts``
# are NOT in this list: they are only present when the record was produced
# from an inference run (e.g. ``eval_epoch_difficulty.py``). Rubric-state-only
# files — such as those emitted by snapshot mergers — omit them. Analysis
# scripts that need scores/rollouts must check for their presence themselves
# (e.g. via ``ep.get("rollouts")``).
REQUIRED_EPOCH_FIELDS = (
    "epoch",
    "requirements",
    "weights",
    "criterion_sources",
)


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_records(input_path: Path) -> list[dict]:
    """Load eval-result JSONL and validate minimum schema."""
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{ln} invalid JSON: {exc}") from exc
            for fld in REQUIRED_RECORD_FIELDS:
                if fld not in rec:
                    raise ValueError(f"{input_path}:{ln} missing field {fld!r}")
            for ep in rec["epochs"]:
                for fld in REQUIRED_EPOCH_FIELDS:
                    if fld not in ep:
                        raise ValueError(
                            f"{input_path}:{ln} epoch={ep.get('epoch')} missing {fld!r}"
                        )
            records.append(rec)
    return records


def discover_epochs(records: Iterable[dict]) -> list[int]:
    """Return sorted list of all epoch ids that appear in the records."""
    s: set[int] = set()
    for rec in records:
        for ep in rec["epochs"]:
            s.add(int(ep["epoch"]))
    return sorted(s)


def filter_idx(epoch_record: dict, source: str) -> list[int]:
    """Return criterion indices in ``requirements`` with the given ``criterion_sources`` value."""
    srcs = epoch_record.get("criterion_sources") or []
    return [i for i, s in enumerate(srcs) if s == source]


def valid_judge_runs(score_runs: list[int]) -> list[int]:
    """Drop None / negative entries (parse failures) from a per-(rollout, criterion) judge-run list."""
    return [s for s in score_runs if s is not None and s >= 0]


def per_rollout_mean_over_criteria(
    epoch_record: dict,
    criterion_indices: list[int],
) -> list[float | None]:
    """For each rollout, unweighted mean of valid scores across the chosen criteria.

    Skips criteria with no valid judge runs. Returns ``None`` for a rollout
    that has no valid score across any of the criteria.
    """
    out: list[float | None] = []
    for per_roll in epoch_record["scores"]:
        vals: list[int] = []
        for k in criterion_indices:
            vals.extend(valid_judge_runs(per_roll[k]))
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def per_rollout_weighted_mean_over_criteria(
    epoch_record: dict,
    criterion_indices: list[int],
) -> list[float | None]:
    """For each rollout, weighted mean of valid scores across the chosen criteria.

    A criterion contributes its (weight * mean-of-valid-judge-runs); criteria
    with no valid judge runs are skipped (their weight excluded from the
    denominator). Returns ``None`` if no criterion contributes for the rollout.
    """
    weights = epoch_record["weights"]
    out: list[float | None] = []
    for per_roll in epoch_record["scores"]:
        num = 0.0
        den = 0.0
        for k in criterion_indices:
            valid = valid_judge_runs(per_roll[k])
            if not valid:
                continue
            sk = sum(valid) / len(valid)
            num += weights[k] * sk
            den += weights[k]
        out.append(num / den if den > 0 else None)
    return out


def per_criterion_mean_over_rollouts(
    epoch_record: dict,
    criterion_indices: list[int],
) -> list[float | None]:
    """For each criterion in ``criterion_indices``, mean of valid scores across all rollouts."""
    out: list[float | None] = []
    for k in criterion_indices:
        vals: list[int] = []
        for per_roll in epoch_record["scores"]:
            vals.extend(valid_judge_runs(per_roll[k]))
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def safe_mean(xs: Iterable[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def safe_stdev(xs: Iterable[float | None]) -> float | None:
    # Sample standard deviation with Bessel's correction (ddof=1),
    # matching the per-group std used to normalize GRPO advantages.
    # ``statistics.stdev`` uses ddof=1 by default (``pstdev`` is the
    # population variant with ddof=0); we rely on that default here.
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def safe_median(xs: Iterable[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# Slim merged-rollout JSONL helpers (one row per rollout, with `step`,
# `criterion_sources`, `weights`, and stringified `rubric_judgement_scores`).
# Used by rubric_score_trajectory.py and any other per-step analysis on the
# output of merge_adaptive_rubric_only_log.py / merge_dataset_snapshots.py.
# ---------------------------------------------------------------------------


def parse_str_list(value):
    """Parse a JSON-list-shaped string field into a Python list.

    Slim-merged rollout rows store some fields (e.g. ``rubric_judgement_scores``,
    ``rubric_parse_success_bool``) as their stringified ``repr``. Returns the
    value unchanged if it's already a list.
    """
    if isinstance(value, str):
        return json.loads(value)
    return value


def per_rollout_weighted_mean_by_source(
    scores: list[float],
    weights: list[int],
    sources: list[str],
    target_sources: list[str],
) -> float | None:
    """Single rollout: weighted mean over criteria whose source ∈ ``target_sources``.

    Skips criteria with score < 0 (judge parse fail). Returns ``None`` if no
    valid criterion contributes.
    """
    target = set(target_sources)
    num = 0.0
    den = 0.0
    for s, w, src in zip(scores, weights, sources):
        if src not in target:
            continue
        if s is None or s < 0:
            continue
        num += w * s
        den += w
    return num / den if den > 0 else None


def ema(xs: list[float], alpha: float) -> list[float]:
    """Exponential moving average with smoothing factor ``alpha`` ∈ (0, 1].

    ``alpha = 1`` returns ``xs`` unchanged; smaller alpha → smoother. ``None``
    entries are skipped (the EMA carries the previous value through gaps).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    out: list[float | None] = []
    cur: float | None = None
    for x in xs:
        if x is None:
            out.append(cur)
            continue
        cur = x if cur is None else alpha * x + (1 - alpha) * cur
        out.append(cur)
    return out
