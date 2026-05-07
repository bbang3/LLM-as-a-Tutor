"""LLM output parsing and prompt formatting utilities.

Handles both directions of text ↔ structured data:
  - **Parsing**: extract scores, decisions, rubrics, etc. from raw LLM output.
  - **Formatting**: build prompt fragments (rubric lists, evaluation summaries).
"""

from __future__ import annotations

import re
import statistics

# ---------------------------------------------------------------------------
# Constants & regex patterns
# ---------------------------------------------------------------------------

_DEFAULT_PARSE: dict[str, str] = {
    "thinking": r"<think>(.*?)</think>",
    "analysis": r"<analysis>(.*?)</analysis>",
    "decision": r"<decision>(.*?)</decision>",
    "constraint": r"<constraint>(.*?)</constraint>",
    "rubric": r"<rubric>(.*?)</rubric>",
    "criterion": r"<criterion>(.*?)</criterion>",
    "importance": r"<importance>(.*?)</importance>",
    "justification": r"<justification>(.*?)</justification>",
}
_RESERVED_PARSE_KEYS = {
    "thinking",
    "decision",
    "analysis",
    "constraint",
    "rubric",
    "criterion",
    "importance",
    "justification",
}
# Match training's _extract_score in verl/llm_tutor/reward_wildchecklists.py:
# 3-level fallback (tag → braced tag → bare line-isolated number), decimal-aware.
_SCORE_TAG_RE = re.compile(r"<score>\s*([-+]?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE | re.DOTALL)
_SCORE_TAG_BRACE_RE = re.compile(
    r"<score>\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*</score>", re.IGNORECASE | re.DOTALL
)
_SCORE_BARE_NUMBER_RE = re.compile(r"(?:^|[\n\r])\s*([-+]?\d+(?:\.\d+)?)\s*(?:[\n\r]|$)")
_PARSE_FAIL_SCORE = -1.0
_SCORE_MIN = -1.0
_SCORE_MAX = 100.0
_JUSTIFICATION_RE = re.compile(r"<justification>(.*?)</justification>", re.DOTALL | re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_SENTENCE_END_CHARS = frozenset('.?!)"\'"\u2019')

_MAX_JUSTIFICATION_CHARS = 300
_MAX_RESPONSE_CHARS = 3000

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def split_thinking(raw: str, thinking_pattern: str | None = None) -> tuple[str, str]:
    """Split raw LLM output into (thinking, response).

    Handles both complete ``<think>...</think>`` blocks and truncated
    outputs where the closing tag is missing.
    """
    pattern = thinking_pattern or _DEFAULT_PARSE["thinking"]
    think_match = re.search(pattern, raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        response = re.sub(pattern, "", raw, flags=re.DOTALL).strip()
    else:
        open_match = re.search(r"<think>", raw, re.IGNORECASE)
        if open_match:
            thinking = raw[open_match.end() :].strip()
            response = raw[: open_match.start()].strip()
        else:
            thinking = ""
            response = raw.strip()
    return thinking, response


def strip_judge_thinking(text: str) -> str:
    """Mirrors verl/llm_tutor/reward_wildchecklists.py::_remove_thinking_from_completion:
    if both ``<think>`` and ``</think>`` are present, take content after the
    LAST ``</think>``; otherwise return the text unchanged (just stripped).
    """
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def parse_judge_output(text: str) -> tuple[float, str]:
    """Extract score and justification from judge output.

    Mirrors verl/llm_tutor/reward_wildchecklists.py:
      * 3-level score-tag fallback (``<score>N</score>`` → ``<score>{N}</score>``
        → bare line-isolated number), decimal-aware.
      * Successful scores are clamped to [-1, 100].
      * Parse failure returns ``-1.0``.
    Justification handling preserves the existing best-effort extraction
    (``<justification>...</justification>``, then open-tag-only, else surrounding
    text).
    """
    score_val: float | None = None
    for pat in (_SCORE_TAG_RE, _SCORE_TAG_BRACE_RE, _SCORE_BARE_NUMBER_RE):
        matches = pat.findall(text)
        if matches:
            score_val = float(matches[-1])
            break

    if score_val is None:
        return _PARSE_FAIL_SCORE, text.strip()

    score = max(_SCORE_MIN, min(_SCORE_MAX, score_val))
    j = _JUSTIFICATION_RE.search(text)
    if j:
        return score, j.group(1).strip()
    open_tag = re.search(r"<justification>", text, re.IGNORECASE)
    if open_tag:
        return score, text[open_tag.end() :].strip()
    score_tag = _SCORE_TAG_RE.search(text)
    if score_tag:
        return score, text[: score_tag.start()].strip()
    return score, text.strip()




def parse_decision(response: str, parse_patterns: dict[str, str] | None = None) -> bool | None:
    """Parse ``<decision>yes/no</decision>``. Returns True, False, or None on failure."""
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("decision")
    if not pat:
        return None
    m = re.search(pat, response, re.DOTALL)
    if not m:
        return None
    text = m.group(1).strip().lower()
    if text in ("yes", "true", "1"):
        return True
    if text in ("no", "false", "0"):
        return False
    return None


def parse_analysis(response: str, parse_patterns: dict[str, str] | None = None) -> str:
    """Extract ``<analysis>`` text from response."""
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("analysis")
    if not pat:
        return ""
    m = re.search(pat, response, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_constraint(response: str, parse_patterns: dict[str, str] | None = None) -> str:
    """Extract ``<constraint>`` text from response."""
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("constraint")
    if not pat:
        return ""
    m = re.search(pat, response, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_constraint_judgment(
    response: str, parse_patterns: dict[str, str] | None = None
) -> tuple[bool | None, str, str, list[dict]]:
    """Parse ``<decision>`` + ``<constraint>`` + optional ``<rubric>`` output.

    Returns ``(needs_constraint, constraint_text, analysis, constraint_rubric_pairs)``.
    ``constraint_rubric_pairs`` contains criteria/importance dicts for the
    constraint only (from Step 5). Empty when decision is NO or rubric is absent.
    When ``needs_constraint`` is None, parsing failed (retry).
    """
    analysis = parse_analysis(response, parse_patterns)
    decision = parse_decision(response, parse_patterns)
    constraint_raw = parse_constraint(response, parse_patterns).strip()

    if decision is None:
        return None, "", analysis, []
    if not decision:
        return False, "", analysis, []
    if not constraint_raw:
        return None, "", analysis, []

    _, rubric_pairs = parse_rubric_output(response, parse_patterns)
    return True, constraint_raw, analysis, rubric_pairs


def parse_rubric_output(
    response: str,
    parse_patterns: dict[str, str] | None = None,
) -> tuple[dict, list[dict]]:
    """Parse metadata + rubric criteria/importance from LLM output."""
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}

    meta: dict[str, str] = {}
    for key, pat in patterns.items():
        if key in _RESERVED_PARSE_KEYS:
            continue
        m = re.search(pat, response, re.DOTALL)
        meta[key] = m.group(1).strip() if m else ""

    rubric_blocks = re.findall(patterns["rubric"], response, re.DOTALL)
    if len(rubric_blocks) != 1:
        return meta, []

    rubric_block = rubric_blocks[0]
    criteria = re.findall(patterns["criterion"], rubric_block, re.DOTALL)
    importances_raw = re.findall(patterns["importance"], rubric_block, re.DOTALL)
    if not criteria or len(criteria) != len(importances_raw):
        return meta, []

    pairs = []
    for crit, imp_raw in zip(criteria, importances_raw):
        try:
            importance = int(imp_raw.strip())
        except ValueError:
            importance = 50
        pairs.append({"rubric": crit.strip(), "importance": importance})
    return meta, pairs


def is_valid_rubric(pairs: list[dict], require_importance: bool = False) -> bool:
    """Check whether parsed rubric pairs are valid."""
    if not pairs:
        return False
    for p in pairs:
        if not p.get("rubric", "").strip():
            return False
        if require_importance:
            imp = p.get("importance")
            if imp is None or not isinstance(imp, int) or not (0 <= imp <= 100):
                return False
    return True


def explain_rubric_failure(
    response: str,
    parse_patterns: dict[str, str] | None = None,
    require_importance: bool = False,
) -> str:
    """Return a human-readable reason why rubric parsing failed."""
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    rubric_pat = patterns.get("rubric")
    criterion_pat = patterns.get("criterion")
    importance_pat = patterns.get("importance")
    if not rubric_pat:
        return "parse.rubric pattern is missing"
    if not criterion_pat:
        return "parse.criterion pattern is missing"
    if not importance_pat:
        return "parse.importance pattern is missing"

    rubric_blocks = re.findall(rubric_pat, response, re.DOTALL)
    if not rubric_blocks:
        return "no <rubric> block found"
    if len(rubric_blocks) != 1:
        return f"expected exactly 1 <rubric> block; found {len(rubric_blocks)}"

    rubric_block = rubric_blocks[0]
    criteria = re.findall(criterion_pat, rubric_block, re.DOTALL)
    importances_raw = re.findall(importance_pat, rubric_block, re.DOTALL)
    if not criteria:
        return "no <criterion> matches found inside <rubric>"
    if not importances_raw:
        return "no <importance> matches found inside <rubric>"
    if len(criteria) != len(importances_raw):
        return f"count mismatch: criterion={len(criteria)} importance={len(importances_raw)}"

    if require_importance:
        for i, raw in enumerate(importances_raw):
            try:
                imp = int(raw.strip())
            except ValueError:
                return f"importance[{i}] not int: {raw!r}"
            if not (0 <= imp <= 100):
                return f"importance[{i}] out of range: {imp}"
    return "unknown parse failure"


def join_prompt_constraint(prompt: str, constraint: str) -> str:
    """Append a constraint to an instruction prompt with appropriate separator."""
    stripped = prompt.rstrip()
    if not stripped:
        return constraint
    if "\n" in stripped:
        sep = "\n"
    elif stripped[-1] in _SENTENCE_END_CHARS:
        sep = " "
    else:
        sep = "\n"
    return stripped + sep + constraint


# ---------------------------------------------------------------------------
# Formatting helpers (structured data → prompt text)
# ---------------------------------------------------------------------------


def format_rubric_criteria(requirements: list[str], weights: list[int]) -> str:
    """Format rubric criteria as a numbered list with weights."""
    lines = []
    for i, (req, w) in enumerate(zip(requirements, weights), 1):
        lines.append(f"{i}. [Weight: {w}] {req}")
    return "\n".join(lines)


def format_evaluation_data(
    requirements: list[str],
    weights: list[int],
    rollout_texts: list[str],
    scores: dict[tuple[int, int, int], list[int]],
    justifications: dict[tuple[int, int, int], list[str]],
    sample_idx: int,
    n_analysis_rollouts: int,
) -> str:
    """Format rollouts + per-criterion scores/justifications for the analysis prompt."""
    lines: list[str] = []
    n_show = min(n_analysis_rollouts, len(rollout_texts))

    for r in range(n_show):
        text = rollout_texts[r]
        if len(text) > _MAX_RESPONSE_CHARS:
            text = text[:_MAX_RESPONSE_CHARS] + "\n... (truncated)"
        lines.append(f"### Response {r + 1}")
        lines.append(f"<response_{r + 1}>")
        lines.append(text)
        lines.append(f"</response_{r + 1}>")
        lines.append("")

    lines.append("### Per-Criterion Evaluation Results")
    lines.append("")
    for k, (req, w) in enumerate(zip(requirements, weights)):
        lines.append(f'#### Criterion {k + 1}: "{req}" (weight: {w})')
        criterion_scores: list[float] = []
        for r in range(n_show):
            run_scores = scores.get((sample_idx, r, k), [])
            run_justs = justifications.get((sample_idx, r, k), [])
            valid = [s for s in run_scores if s >= 0]
            if valid:
                mean_score = sum(valid) / len(valid)
                criterion_scores.append(mean_score)
                first_just = run_justs[0] if run_justs else ""
                if len(first_just) > _MAX_JUSTIFICATION_CHARS:
                    first_just = first_just[:_MAX_JUSTIFICATION_CHARS] + "..."
                lines.append(f"- Response {r + 1} [Score: {mean_score:.0f}]: {first_just}")
            else:
                lines.append(f"- Response {r + 1}: No valid score")
        if criterion_scores:
            mean = statistics.mean(criterion_scores)
            std = statistics.stdev(criterion_scores) if len(criterion_scores) > 1 else 0.0
            lines.append(
                f"Score stats: mean={mean:.1f}, min={min(criterion_scores):.0f}, "
                f"max={max(criterion_scores):.0f}, std={std:.1f}"
            )
        lines.append("")

    return "\n".join(lines)
