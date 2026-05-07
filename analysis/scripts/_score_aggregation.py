"""Per-rollout score aggregation shared by eval / rescore / analysis scripts.

The eval pipeline stores raw judge results as a 3-D tensor per condition:

    scores[r][k][j]  = judge score for (rollout r, criterion k, judge run j)

Most analyses want a *scalar per rollout* — the weighted mean over criteria,
optionally filtered to a single ``criterion_sources`` bucket (e.g. "general").
``aggregate_per_rollout`` does that aggregation once so consumers don't each
re-implement the boundary cases (parse-fail sentinel, zero valid judges,
multiple ``n_judge_runs``).

Schema convention saved into eval-output JSONs (so analysis scripts can read
directly without recomputing):

    rollout_scores_<cond>_all       — list[float|None], length n_rollouts
    rollout_scores_<cond>_general   — list[float|None]
    rollout_scores_<cond>_constraint — list[float|None]   (only when constraint criteria exist)

where ``<cond>`` is "original" or "constraint" (= treated condition: appended
constraint or rewrite). ``None`` entries indicate the rollout had no valid
criterion to aggregate (e.g. all judge calls returned parse_fail).
"""

from __future__ import annotations


def _per_rollout_weighted_mean(
    scores_per_criterion: list,   # [n_criteria][n_judge_runs]
    weights: list,                # [n_criteria]
    sources: list | None,         # [n_criteria] or None
    keep_source: str | None,      # restrict to this criterion_sources value, or None for all
) -> float | None:
    """Weighted mean over kept criteria. None if no valid value."""
    num = 0.0
    den = 0.0
    any_valid = False
    n_k = min(len(scores_per_criterion), len(weights))
    for k in range(n_k):
        if keep_source is not None:
            if sources is None:
                raise ValueError("keep_source requires sources")
            if sources[k] != keep_source:
                continue
        judge_runs = scores_per_criterion[k] or []
        valid = [s for s in judge_runs if s is not None and s >= 0]
        if not valid:
            continue
        crit_score = sum(valid) / len(valid)
        w = float(weights[k])
        num += crit_score * w
        den += w
        any_valid = True
    if not any_valid or den <= 0:
        return None
    return num / den


def aggregate_per_rollout(
    scores_tensor: list,   # [n_rollouts][n_criteria][n_judge_runs]
    weights: list,         # [n_criteria]
    sources: list,         # [n_criteria]; pass [] / None to skip source-split buckets
) -> dict[str, list]:
    """Per-rollout weighted-mean aggregates split by ``criterion_sources``.

    Returns dict with up to three keys: ``all``, ``general``, ``constraint``.
    Buckets with no matching criteria are omitted. Each value is a list of
    length ``n_rollouts`` (one scalar or None per rollout).
    """
    n_rollouts = len(scores_tensor)
    n_criteria = len(weights)
    out: dict[str, list] = {"all": [None] * n_rollouts}
    has_general = bool(sources) and any(s == "general" for s in sources)
    has_constraint = bool(sources) and any(s == "constraint" for s in sources)
    if has_general:
        out["general"] = [None] * n_rollouts
    if has_constraint:
        out["constraint"] = [None] * n_rollouts
    for r in range(n_rollouts):
        ros = scores_tensor[r]
        if not ros:
            continue
        out["all"][r] = _per_rollout_weighted_mean(ros, weights, None, None)
        if has_general:
            out["general"][r] = _per_rollout_weighted_mean(ros, weights, sources, "general")
        if has_constraint:
            out["constraint"][r] = _per_rollout_weighted_mean(ros, weights, sources, "constraint")
    return out
