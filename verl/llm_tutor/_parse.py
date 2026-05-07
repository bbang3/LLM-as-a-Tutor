"""Parsing helpers copied from ``data_generation/parse_utils.py``.

Used by OnlineDataGenerator so the verl package has no sys.path dependency
on the offline data_generation directory. Keep these in sync if parse_utils
changes upstream.
"""

from __future__ import annotations

import json
import re

_DEFAULT_PARSE: dict[str, str] = {
    "thinking": r"<think>(.*?)</think>",
    "analysis": r"<analysis>(.*?)</analysis>",
    "decision": r"<decision>(.*?)</decision>",
    "constraint": r"<constraint>(.*?)</constraint>",
    "rewrite": r"<rewrite>(.*?)</rewrite>",
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
    "rewrite",
    "rubric",
    "criterion",
    "importance",
    "justification",
}
_SENTENCE_END_CHARS = frozenset('.?!)"\'"\u2019')


def split_thinking(raw: str, thinking_pattern: str | None = None) -> tuple[str, str]:
    pattern = thinking_pattern or _DEFAULT_PARSE["thinking"]
    think_match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        response = re.sub(pattern, "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        open_match = re.search(r"<think>", raw, re.IGNORECASE)
        if open_match:
            thinking = raw[open_match.end() :].strip()
            response = raw[: open_match.start()].strip()
        else:
            thinking = ""
            response = raw.strip()
    return thinking, response


def parse_decision(response: str, parse_patterns: dict[str, str] | None = None) -> bool | None:
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("decision")
    if not pat:
        return None
    m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    text = m.group(1).strip().lower()
    if text in ("yes", "true", "1"):
        return True
    if text in ("no", "false", "0"):
        return False
    return None


def parse_analysis(response: str, parse_patterns: dict[str, str] | None = None) -> str:
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("analysis")
    if not pat:
        return ""
    m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_constraint(response: str, parse_patterns: dict[str, str] | None = None) -> str:
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("constraint")
    if not pat:
        return ""
    m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_rewrite(response: str, parse_patterns: dict[str, str] | None = None) -> str:
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}
    pat = patterns.get("rewrite")
    if not pat:
        return ""
    m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_rubric_output(
    response: str,
    parse_patterns: dict[str, str] | None = None,
) -> tuple[dict, list[dict]]:
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}

    meta: dict[str, str] = {}
    for key, pat in patterns.items():
        if key in _RESERVED_PARSE_KEYS:
            continue
        m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        meta[key] = m.group(1).strip() if m else ""

    rubric_blocks = re.findall(patterns["rubric"], response, re.DOTALL | re.IGNORECASE)
    if len(rubric_blocks) != 1:
        return meta, []

    rubric_block = rubric_blocks[0]
    criteria = re.findall(patterns["criterion"], rubric_block, re.DOTALL | re.IGNORECASE)
    importances_raw = re.findall(patterns["importance"], rubric_block, re.DOTALL | re.IGNORECASE)
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


def parse_rubric_output_lenient(
    response: str,
    parse_patterns: dict[str, str] | None = None,
) -> tuple[dict, list[dict]]:
    """Best-effort rubric parse used as a last-resort fallback after retries.

    Differs from ``parse_rubric_output`` by:
    - Accepting count mismatches (zip truncates to the shorter list).
    - Using the first ``<rubric>`` block when multiple are present.
    - Skipping pairs with empty criterion text.
    - Clamping importance into [0, 100] instead of dropping OOR values.
    """
    patterns = {**_DEFAULT_PARSE, **(parse_patterns or {})}

    meta: dict[str, str] = {}
    for key, pat in patterns.items():
        if key in _RESERVED_PARSE_KEYS:
            continue
        m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        meta[key] = m.group(1).strip() if m else ""

    rubric_blocks = re.findall(patterns["rubric"], response, re.DOTALL | re.IGNORECASE)
    if not rubric_blocks:
        return meta, []

    rubric_block = rubric_blocks[0]
    criteria = re.findall(patterns["criterion"], rubric_block, re.DOTALL | re.IGNORECASE)
    importances_raw = re.findall(patterns["importance"], rubric_block, re.DOTALL | re.IGNORECASE)

    pairs: list[dict] = []
    for crit, imp_raw in zip(criteria, importances_raw):
        crit_text = crit.strip()
        if not crit_text:
            continue
        try:
            importance = int(imp_raw.strip())
        except ValueError:
            importance = 50
        importance = max(0, min(100, importance))
        pairs.append({"rubric": crit_text, "importance": importance})
    return meta, pairs


def is_valid_rubric(pairs: list[dict], require_importance: bool = False) -> bool:
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


def parse_constraint_judgment(
    response: str, parse_patterns: dict[str, str] | None = None
) -> tuple[bool | None, str, str, list[dict]]:
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


def parse_constraint_rewrite_judgment(
    response: str, parse_patterns: dict[str, str] | None = None
) -> tuple[bool | None, str, str]:
    """Parse a constraint-rewrite judgment.

    Returns ``(decision, rewritten_instruction, analysis)``:
      * ``decision=False`` — no rewrite needed; ``rewritten_instruction`` empty.
      * ``decision=True`` with non-empty ``rewritten_instruction`` — apply rewrite.
      * decision missing, or yes-without-rewrite → ``(None, "", analysis)`` (retry).
    """
    analysis = parse_analysis(response, parse_patterns)
    decision = parse_decision(response, parse_patterns)
    rewrite_text = parse_rewrite(response, parse_patterns).strip()

    if decision is None:
        return None, "", analysis
    if not decision:
        return False, "", analysis
    if not rewrite_text:
        return None, "", analysis
    return True, rewrite_text, analysis


def parse_json_filter_output(response: str) -> dict | None:
    """Extract a JSON object from a judge response.

    Tries a ```json fenced block first, then falls back to the outermost
    brace pair. Returns ``None`` on parse failure.
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def join_prompt_constraint(prompt: str, constraint: str) -> str:
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
