"""Metric-based saturation gate for the ``constraint_metric`` data-generation mode.

This is the *metric* counterpart to the LLM ``<decision>`` judge used by the
``constraint`` mode. Instead of asking an LLM whether two rollouts are
indistinguishable, it scores ``n`` policy rollouts against the prompt's
**general rubric** with the same per-criterion ``judge`` protocol the
training reward uses, computes the cross-rollout standard deviation of the
per-rollout weighted-mean score, and flags a prompt as *saturated* when that
std falls below a threshold. Saturated prompts then receive an unconditional
constraint append (via ``_gen_constraint_always``).

The module is deliberately transport-free: it builds judge inputs and parses /
aggregates judge outputs, but the actual HTTP calls are made by the caller's
existing ``_generate_with_retry_keyed`` path (generator == judge in this
setup), so no new backend, config threading, or concurrency handling is
introduced here.

Scoring fidelity notes:
- Scores stay on the native 0-100 rubric scale (the training reward divides by
  100 at the very end). The threshold is therefore interpreted on 0-100, the
  same scale as the per-prompt rollout-std of the base model (empirically ~6).
- Per-criterion parse / clamp mirrors ``verl/llm_tutor/reward_wildchecklists.py``
  (``_extract_score`` / ``_clamp_reward_score``): parse failure ->
  ``parse_fail_score`` (0.0), clamp to ``[min_score, max_score]`` = ``[-1, 100]``.
- The judge call goes through the generator backend, so it uses the
  generator's configured sampling params (``data_generator.temperature`` etc.),
  not the reward path's ``temperature=0``. For a baseline gate this is fine;
  set ``data_generator.temperature`` low if tighter determinism is wanted.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Hashable

# Key into the flat judge-input dict: (sample position, rollout index, criterion index).
ScoreKey = tuple[int, int, int]

_SCORE_TAG = re.compile(r"<score>\s*([-+]?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE | re.DOTALL)
_SCORE_TAG_BRACE = re.compile(r"<score>\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*</score>", re.IGNORECASE | re.DOTALL)
_BASE_NUMBER = re.compile(r"(?:^|[\n\r])\s*([-+]?\d+(?:\.\d+)?)\s*(?:[\n\r]|$)")


def _strip_thinking(text: str) -> str:
    """Drop a trailing ``<think>...</think>`` block, mirroring the judge path."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def extract_score(judge_completion: str) -> float | None:
    """Parse the judge's ``<score>`` value; mirrors reward_wildchecklists._extract_score."""
    body = _strip_thinking(judge_completion)
    for pattern in (_SCORE_TAG, _SCORE_TAG_BRACE, _BASE_NUMBER):
        matches = pattern.findall(body)
        if matches:
            return float(matches[-1])
    return None


def clamp_score(
    score: float,
    parse_fail_score: float = 0.0,
    min_score: float | None = -1.0,
    max_score: float | None = 100.0,
) -> float:
    """Clamp a parsed score; mirrors reward_wildchecklists._clamp_reward_score."""
    if not math.isfinite(score):
        return float(parse_fail_score)
    if min_score is not None:
        score = max(score, float(min_score))
    if max_score is not None:
        score = min(score, float(max_score))
    return float(score)


def parse_score_strict(raw: str) -> float | None:
    """Strict parse_fn for ``_generate_with_retry_keyed``: parsed score or None."""
    score = extract_score(raw)
    if score is None or not math.isfinite(score):
        return None
    return clamp_score(score)


def parse_score_fallback(raw: str, parse_fail_score: float = 0.0) -> float:
    """Lenient fallback after retries: keep the sample, score = parse_fail_score.

    Matches the reward path, which scores an unparseable criterion as
    ``parse_fail_score`` rather than dropping it (a dropped criterion would
    silently change a rollout's weighted mean and corrupt the std estimate).
    """
    return float(parse_fail_score)


def build_score_inputs(
    prompts: list[str],
    rollouts: list[list[str]],
    general_rubrics: dict[int, list[dict]],
    eligible: list[int],
    template: str,
    min_valid_rollouts: int = 2,
) -> tuple[dict[ScoreKey, str], dict[int, dict]]:
    """Build the flat ``{(pos, r_idx, c_idx): judge_prompt}`` request dict.

    ``general_rubrics[pos]`` is a list of ``{"rubric": <criterion>, "importance": <int>}``
    (the shape produced by ``parse_rubric_output``). A prompt is included only
    when it has >= ``min_valid_rollouts`` rollouts and a non-empty general
    rubric. Returns the request dict plus per-position metadata
    (``n_rollouts``, ``weights``, ``criteria``) used to aggregate.
    """
    inputs: dict[ScoreKey, str] = {}
    meta: dict[int, dict] = {}
    for pos in eligible:
        responses = rollouts[pos] if pos < len(rollouts) else []
        pairs = general_rubrics.get(pos) or []
        if len(responses) < min_valid_rollouts or not pairs:
            continue
        weights = [float(p.get("importance", 50)) for p in pairs]
        meta[pos] = {"n_rollouts": len(responses), "weights": weights, "criteria": len(pairs)}
        for r_idx, response in enumerate(responses):
            for c_idx, pair in enumerate(pairs):
                inputs[(pos, r_idx, c_idx)] = template.format(
                    instruction=prompts[pos],
                    response=response,
                    requirement=pair["rubric"],
                )
    return inputs, meta


def _weighted_mean(scores: list[float], weights: list[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights, strict=True)) / total_w


def aggregate_saturation(
    score_results: dict[ScoreKey, float],
    meta: dict[int, dict],
    threshold: float,
    parse_fail_score: float = 0.0,
) -> tuple[set[int], dict[int, dict]]:
    """Aggregate per-criterion scores into per-prompt rollout-std and gate on it.

    For each position: per-rollout weighted mean over criteria, then the
    population std (ddof=0, matching the GRPO group std) across rollouts. A
    position is *saturated* when ``std < threshold``. Returns the saturated
    position set and a per-position metrics dict (``mean``, ``std``,
    ``n_rollouts``, ``n_criteria``, ``rollout_scores``, ``saturated``).
    """
    saturated: set[int] = set()
    metrics: dict[int, dict] = {}
    for pos, info in meta.items():
        n_roll = info["n_rollouts"]
        weights = info["weights"]
        n_crit = len(weights)
        rollout_scores: list[float] = []
        for r_idx in range(n_roll):
            crit_scores = [float(score_results.get((pos, r_idx, c_idx), parse_fail_score)) for c_idx in range(n_crit)]
            rollout_scores.append(_weighted_mean(crit_scores, weights))
        std = float(statistics.pstdev(rollout_scores)) if len(rollout_scores) > 1 else 0.0
        mean = float(statistics.fmean(rollout_scores)) if rollout_scores else 0.0
        is_sat = std < threshold
        if is_sat:
            saturated.add(pos)
        metrics[pos] = {
            "mean": mean,
            "std": std,
            "n_rollouts": n_roll,
            "n_criteria": n_crit,
            "rollout_scores": rollout_scores,
            "saturated": is_sat,
        }
    return saturated, metrics
