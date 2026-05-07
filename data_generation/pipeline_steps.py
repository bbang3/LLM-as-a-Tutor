"""Reusable pipeline building blocks: rollout generation, rubric generation, scoring.

Each function orchestrates batched LLM calls with automatic retry logic.
"""

from __future__ import annotations

from parse_utils import (
    _ANALYSIS_RE,
    format_evaluation_data,
    format_rubric_criteria,
    is_valid_rubric,
    parse_judge_output,
    parse_rubric_output,
    split_thinking,
    strip_judge_thinking,
)
from vllm_dp import LLMBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRY = 10

# Sentinel sent to the judge in place of an empty/truncated policy response.
# Mirrors verl/llm_tutor/reward_wildchecklists.py so the judge
# parse-fails on these and the rollout takes parse_fail_score (-1).
INVALID_RESPONSE_SENTINEL = "<invalid_response>"

# ---------------------------------------------------------------------------
# Step: Generate rollouts (policy model)
# ---------------------------------------------------------------------------


def generate_rollouts(
    samples: list[dict],
    model: LLMBase,
    tokenizer,
    policy_cfg: dict,
    n_rollouts: int,
    parse_patterns: dict,
    min_valid: int = 1,
    pad_invalid_with_sentinel: bool = False,
) -> list[list[str]]:
    """Generate *n_rollouts* responses per sample using the policy model.

    A sample is retried until at least *min_valid* non-empty responses are
    produced in a single attempt. Default ``min_valid=1`` keeps the historical
    lenient behavior; pass a larger value (e.g. 2) when the caller requires
    multiple base responses per prompt.

    When ``pad_invalid_with_sentinel`` is True, accepted samples keep all
    ``n_rollouts`` responses and any empty/truncated ones are replaced with
    :data:`INVALID_RESPONSE_SENTINEL` instead of being filtered out. Use this
    for scoring pipelines that need to mirror training reward (where empty
    rollouts go to the judge and parse-fail to score=-1). Default False
    preserves the legacy behavior used by rubric-generation callers.
    """
    sp_kwargs = dict(policy_cfg["sampling"])
    sp_kwargs["n"] = n_rollouts
    enable_thinking = policy_cfg.get("enable_thinking", False)
    thinking_pat = parse_patterns.get("thinking")

    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for item in samples
    ]

    N = len(samples)
    rollouts: list[list[str] | None] = [None] * N
    failed = list(range(N))

    for attempt in range(MAX_RETRY + 1):
        if not failed:
            break
        batch = [prompt_texts[i] for i in failed]
        outputs = model.generate(batch, sp_kwargs)
        still_failed = []
        for k, i in enumerate(failed):
            texts = outputs[k]
            if pad_invalid_with_sentinel:
                # Mirror training policy-side preprocessing exactly
                # (verl/llm_tutor/reward_wildchecklists.py): if both
                # <think> tags are present, take content after the LAST
                # </think>; <think> without </think> means the rollout was
                # truncated → sentinel; otherwise pass the raw text through.
                responses = []
                for t in texts:
                    if "<think>" in t:
                        if "</think>" in t:
                            responses.append(t.split("</think>")[-1].strip())
                        else:
                            responses.append(INVALID_RESPONSE_SENTINEL)
                    else:
                        responses.append(t.strip())
                # Truncated/empty responses don't count toward retry-decision
                # validity; sentinel-padded ones are kept in the final list.
                valid = [
                    r for r in responses if r and r != INVALID_RESPONSE_SENTINEL
                ]
            else:
                responses = [split_thinking(t, thinking_pat)[1].strip() for t in texts]
                valid = [r for r in responses if r]

            if len(valid) >= min_valid:
                if pad_invalid_with_sentinel:
                    rollouts[i] = responses[:n_rollouts]
                else:
                    rollouts[i] = valid[:n_rollouts]
            else:
                still_failed.append(i)
        failed = still_failed
        if failed and attempt < MAX_RETRY:
            print(f"  Rollout retry {attempt + 1}/{MAX_RETRY}: {len(failed)} samples")

    for i in range(N):
        if rollouts[i] is None:
            rollouts[i] = []

    return rollouts  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step: Generate rubrics from rollout responses (judge model)
# ---------------------------------------------------------------------------


def generate_rubrics(
    samples: list[dict],
    rollouts: list[list[str]],
    judge_model: LLMBase,
    judge_tokenizer,
    judge_cfg: dict,
    rubric_gen_template: str,
    parse_patterns: dict,
    include_weight: bool,
    use_responses: bool = True,
) -> tuple[list[list[str]], list[list[int]]]:
    """Generate rubrics from the instruction (and optionally 2 rollout responses).

    When ``use_responses`` is True, the template is formatted with
    ``instruction``, ``response_1``, ``response_2`` and samples with fewer
    than 2 rollouts are skipped (original rubrics preserved).
    When False, only ``instruction`` is passed — the template must not
    reference response fields, and all samples are eligible.
    Returns ``(requirements_list, weights_list)`` for all N samples.
    """
    N = len(samples)
    sp_kwargs = dict(judge_cfg["sampling"])
    sp_kwargs["n"] = 1
    enable_thinking = judge_cfg.get("enable_thinking", False)
    thinking_pat = parse_patterns.get("thinking")

    new_requirements: list[list[str]] = [list(s.get("requirements", [])) for s in samples]
    new_weights: list[list[int]] = [list(s.get("weights", [])) for s in samples]

    gen_indices: list[int] = []
    gen_prompt_texts: list[str] = []
    for i, (sample, sample_rollouts) in enumerate(zip(samples, rollouts)):
        if use_responses and len(sample_rollouts) < 2:
            continue
        gen_indices.append(i)
        # ``constraint`` is forwarded whenever present so templates that split
        # instruction + constraint (e.g. constraint-anchored adaptive rubric)
        # can render it. Templates that do not reference ``{constraint}``
        # ignore the extra kwarg.
        fmt_kwargs = {
            "instruction": sample["prompt"],
            "constraint": sample.get("constraint", ""),
        }
        if use_responses:
            fmt_kwargs["response_1"] = sample_rollouts[0]
            fmt_kwargs["response_2"] = sample_rollouts[1]
        content = rubric_gen_template.format(**fmt_kwargs)
        gen_prompt_texts.append(
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    if not gen_indices:
        print("  No samples have enough rollouts for rubric generation.")
        return new_requirements, new_weights

    print(f"  Generating rubrics for {len(gen_indices)} samples")

    failed = list(range(len(gen_indices)))
    for attempt in range(MAX_RETRY + 1):
        if not failed:
            break
        batch = [gen_prompt_texts[j] for j in failed]
        outputs = judge_model.generate(batch, sp_kwargs)
        still_failed = []
        for pos, j in enumerate(failed):
            raw = outputs[pos][0]
            _, resp = split_thinking(raw, thinking_pat)
            _, pairs = parse_rubric_output(resp, parse_patterns)
            if is_valid_rubric(pairs, require_importance=include_weight):
                idx = gen_indices[j]
                new_requirements[idx] = [p["rubric"] for p in pairs]
                new_weights[idx] = [p["importance"] for p in pairs]
            else:
                still_failed.append(j)
                if attempt == MAX_RETRY:
                    idx = gen_indices[j]
                    print(
                        f"  [Rubric gen failure] idx={idx}\n"
                        f"  [Prompt] {samples[idx]['prompt'][:200]}\n"
                        f"  [Raw output] {raw[:500]}"
                    )
        failed = still_failed
        if failed and attempt < MAX_RETRY:
            print(f"  Rubric gen retry {attempt + 1}/{MAX_RETRY}: {len(failed)} samples")

    n_ok = sum(1 for idx in gen_indices if new_requirements[idx] != list(samples[idx].get("requirements", [])))
    n_fallback = len(gen_indices) - n_ok
    print(f"  Rubric gen: {n_ok} generated, {n_fallback} fell back to original")
    return new_requirements, new_weights


# ---------------------------------------------------------------------------
# Step: Score rollouts against rubric criteria (judge model)
# ---------------------------------------------------------------------------


def score_rollouts(
    samples: list[dict],
    rollouts: list[list[str]],
    judge_model: LLMBase,
    judge_tokenizer,
    judge_cfg: dict,
    judge_template: str,
    parse_patterns: dict,
    n_judge_runs: int,
) -> tuple[
    dict[tuple[int, int, int], list[int]],
    dict[tuple[int, int, int], list[str]],
]:
    """Score each (sample, rollout, criterion) triple *n_judge_runs* times.

    Returns::

        scores:         (i, r, k) -> list[int]  of length n_judge_runs
        justifications: (i, r, k) -> list[str]  of length n_judge_runs
    """
    judge_sp_kwargs = dict(judge_cfg["sampling"])
    judge_sp_kwargs["n"] = n_judge_runs
    enable_thinking = judge_cfg.get("enable_thinking", False)

    tasks: list[tuple[int, int, int]] = []
    task_contents: list[str] = []
    for i, (sample, sample_rollouts) in enumerate(zip(samples, rollouts)):
        for r, rollout in enumerate(sample_rollouts):
            for k, criterion in enumerate(sample["requirements"]):
                content = judge_template.format(
                    instruction=sample["prompt"],
                    response=rollout,
                    requirement=criterion,
                )
                tasks.append((i, r, k))
                task_contents.append(content)

    judge_prompts = [
        judge_tokenizer.apply_chat_template(
            [{"role": "user", "content": c}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for c in task_contents
    ]

    scores: dict[tuple[int, int, int], list[int]] = {key: [] for key in tasks}
    justifications: dict[tuple[int, int, int], list[str]] = {key: [] for key in tasks}

    if not tasks:
        print("  No scoring tasks.")
        return scores, justifications

    print(f"  Total scoring tasks: {len(tasks)}  (n_judge_runs={n_judge_runs})")
    outputs = judge_model.generate(judge_prompts, judge_sp_kwargs)

    parsed_ok = 0
    for idx, key in enumerate(tasks):
        for raw in outputs[idx]:
            # Match training's _remove_thinking_from_completion exactly.
            response = strip_judge_thinking(raw)
            score, justification = parse_judge_output(response)
            scores[key].append(score)
            justifications[key].append(justification)
            if score >= 0:
                parsed_ok += 1

    total = len(tasks) * n_judge_runs
    print(f"  Parsed {parsed_ok}/{total} valid scores ({parsed_ok / max(total, 1) * 100:.1f}%)")
    return scores, justifications


# ---------------------------------------------------------------------------
# Step: Analyze rubric quality (judge model)
# ---------------------------------------------------------------------------


def analyze_rubric_quality(
    samples: list[dict],
    rollouts: list[list[str]],
    scores: dict[tuple[int, int, int], list[int]],
    justifications: dict[tuple[int, int, int], list[str]],
    judge_model: LLMBase,
    judge_tokenizer,
    judge_cfg: dict,
    analyze_template: str,
    parse_patterns: dict,
    n_analysis_rollouts: int,
) -> list[str | None]:
    """Analyze rubric quality for each sample.

    Returns a list of analysis text strings (None for ineligible samples).
    """
    N = len(samples)
    sp_kwargs = dict(judge_cfg["sampling"])
    sp_kwargs["n"] = 1
    enable_thinking = judge_cfg.get("enable_thinking", False)
    thinking_pat = parse_patterns.get("thinking")

    gen_indices: list[int] = []
    gen_prompt_texts: list[str] = []
    for i, sample in enumerate(samples):
        reqs = sample.get("requirements", [])
        weights = sample.get("weights", [])
        if not reqs or not rollouts[i]:
            continue
        gen_indices.append(i)

        rubric_text = format_rubric_criteria(reqs, weights)
        eval_data = format_evaluation_data(
            reqs,
            weights,
            rollouts[i],
            scores,
            justifications,
            i,
            n_analysis_rollouts,
        )
        content = analyze_template.format(
            instruction=sample["prompt"],
            rubric_criteria=rubric_text,
            evaluation_data=eval_data,
        )
        gen_prompt_texts.append(
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    analyses_list: list[str | None] = [None] * N

    if not gen_indices:
        print("  No samples eligible for rubric analysis.")
        return analyses_list

    print(f"  Analyzing rubric quality for {len(gen_indices)} samples")

    failed = list(range(len(gen_indices)))
    for attempt in range(MAX_RETRY + 1):
        if not failed:
            break
        batch = [gen_prompt_texts[j] for j in failed]
        outputs = judge_model.generate(batch, sp_kwargs)
        still_failed = []
        for pos, j in enumerate(failed):
            raw = outputs[pos][0]
            _, resp = split_thinking(raw, thinking_pat)
            idx = gen_indices[j]

            analysis_match = _ANALYSIS_RE.search(resp)
            analysis_text = analysis_match.group(1).strip() if analysis_match else ""

            if analysis_text:
                analyses_list[idx] = analysis_text
            else:
                still_failed.append(j)
                if attempt == MAX_RETRY:
                    analyses_list[idx] = resp.strip() or "(parse failure -- using raw output)"
                    print(f"  [Analysis parse failure] idx={idx}\n" f"  [Raw output] {raw[:500]}")
        failed = still_failed
        if failed and attempt < MAX_RETRY:
            print(f"  Analysis retry {attempt + 1}/{MAX_RETRY}: {len(failed)} samples")

    n_ok = sum(1 for a in analyses_list if a is not None)
    print(f"  Analysis done: {n_ok}/{len(gen_indices)} parsed")
    return analyses_list


# ---------------------------------------------------------------------------
# Step: Model-based rubric filtering
# ---------------------------------------------------------------------------


def filter_rubrics_with_model(
    samples: list[dict],
    requirements: list[list[str]],
    weights: list[list[int]],
    judge_model: LLMBase,
    judge_tokenizer,
    judge_cfg: dict,
    filter_template_str: str,
    parse_patterns: dict,
) -> tuple[list[list[str]], list[list[int]], list[dict]]:
    """Filter rubric criteria using a judge model (model-based filtering).

    For each sample, renders the filter template with the instruction and
    criteria, then asks the judge model to keep/drop each criterion and
    assign fresh importance weights.

    Returns ``(filtered_requirements, filtered_weights, filter_info)`` for all
    N samples. ``filter_info[i]`` is a dict with keys:
      - ``raw_output``: last raw generator output (str)
      - ``parsed``: parsed JSON dict (or None on parse failure)
      - ``kept_ids`` / ``dropped_ids``: 1-indexed criterion ids
      - ``kept_requirements`` / ``dropped_requirements``: criterion texts
      - ``kept_importances``: fresh importances assigned by the judge
      - ``parse_ok``: whether parsing succeeded
    """
    import jinja2
    from filter_rubric_v4_with_model import (
        interpret_filter_output,
        parse_json_filter_output,
    )

    N = len(samples)
    template = jinja2.Template(filter_template_str, keep_trailing_newline=True)

    sp_kwargs = dict(judge_cfg["sampling"])
    sp_kwargs["n"] = 1
    enable_thinking = judge_cfg.get("enable_thinking", False)
    thinking_pat = parse_patterns.get("thinking")

    # Build prompts
    prompt_texts: list[str] = []
    for i in range(N):
        content = template.render(
            instruction=samples[i]["prompt"],
            requirements=requirements[i],
        )
        prompt_texts.append(
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    # Generate with retries
    filtered_reqs: list[list[str]] = [list(r) for r in requirements]
    filtered_weights: list[list[int]] = [list(w) for w in weights]
    pending = list(range(N))
    parsed_outputs: list[dict | None] = [None] * N
    raw_outputs: list[str] = [""] * N

    for attempt in range(MAX_RETRY + 1):
        if not pending:
            break
        batch = [prompt_texts[i] for i in pending]
        outputs = judge_model.generate(batch, sp_kwargs)

        still_pending: list[int] = []
        for pos, i in enumerate(pending):
            raw = outputs[pos][0]
            _, resp = split_thinking(raw, thinking_pat)
            raw_outputs[i] = resp
            parsed = parse_json_filter_output(resp)
            if parsed is not None:
                parsed_outputs[i] = parsed
            else:
                still_pending.append(i)
        pending = still_pending
        if pending and attempt < MAX_RETRY:
            print(f"  Filter retry {attempt + 1}/{MAX_RETRY}: {len(pending)} parse failures")

    # Apply filter results
    n_filtered = 0
    filter_info: list[dict] = []
    for i in range(N):
        parsed = parsed_outputs[i]
        info: dict = {
            "raw_output": raw_outputs[i],
            "parsed": parsed,
            "parse_ok": parsed is not None,
            "original_requirements": list(requirements[i]),
            "original_weights": list(weights[i]),
            "kept_ids": [],
            "dropped_ids": [],
            "kept_requirements": [],
            "dropped_requirements": [],
            "kept_importances": [],
        }
        if parsed is None:
            # Parse failure: keep all original criteria
            info["kept_ids"] = list(range(1, len(requirements[i]) + 1))
            info["kept_requirements"] = list(requirements[i])
            info["kept_importances"] = list(weights[i])
            filter_info.append(info)
            continue
        kept, dropped_ids = interpret_filter_output(parsed, len(requirements[i]))
        new_reqs: list[str] = []
        new_weights: list[int] = []
        kept_ids: list[int] = []
        for k in kept:
            idx = k["id"] - 1
            if idx < len(requirements[i]):
                new_reqs.append(requirements[i][idx])
                new_weights.append(k["importance"])
                kept_ids.append(k["id"])
        info["kept_ids"] = kept_ids
        info["dropped_ids"] = list(dropped_ids)
        info["kept_requirements"] = list(new_reqs)
        info["kept_importances"] = list(new_weights)
        info["dropped_requirements"] = [requirements[i][j - 1] for j in dropped_ids if j - 1 < len(requirements[i])]
        if new_reqs:
            filtered_reqs[i] = new_reqs
            filtered_weights[i] = new_weights
            n_filtered += 1
        filter_info.append(info)

    n_ok = sum(1 for p in parsed_outputs if p is not None)
    print(f"  Model filter: {n_ok}/{N} parsed, {n_filtered} filtered")
    return filtered_reqs, filtered_weights, filter_info


# ---------------------------------------------------------------------------
# Step: Adaptive rubric dedup filtering against general/constraint rubrics
# ---------------------------------------------------------------------------


def filter_adaptive_rubrics_by_dedup(
    samples: list[dict],
    adaptive_requirements: list[list[str]],
    adaptive_weights: list[list[int]],
    general_requirements: list[list[str]],
    constraint_requirements: list[list[str]],
    judge_model: LLMBase,
    judge_tokenizer,
    judge_cfg: dict,
    dedup_template_str: str,
    parse_patterns: dict,
) -> tuple[list[list[str]], list[list[int]], list[dict]]:
    """Drop adaptive criteria that semantically duplicate general/constraint criteria.

    For each sample the judge is given the instruction plus three rubric
    groups (general, constraint, adaptive) and asked to decide, per adaptive
    criterion, whether it is a duplicate of any general or constraint
    criterion. Duplicates are dropped and the matching reference criterion
    is recorded.

    Samples with no adaptive criteria are skipped; samples where both
    general and constraint lists are empty are also skipped (nothing to
    dedupe against). Kept adaptive criteria retain their original weights.

    Returns ``(filtered_adaptive_requirements, filtered_adaptive_weights,
    dedup_info)`` for all N samples. ``dedup_info[i]`` is a dict with keys:
      - ``parse_ok``: whether JSON parsing succeeded
      - ``eligible``: whether the sample was sent to the judge
      - ``raw_output``: last raw response text (empty when not eligible)
      - ``parsed``: parsed JSON dict (or None on parse failure / not eligible)
      - ``original_adaptive_requirements`` / ``original_adaptive_weights``
      - ``kept_ids`` / ``kept_requirements`` / ``kept_weights``
        (1-indexed adaptive ids)
      - ``dropped``: list of dicts, one per dropped adaptive criterion::

            {
              "adaptive_id":      <int, 1-indexed>,
              "adaptive_text":    <str>,
              "matched_source":   "general" | "constraint" | None,
              "matched_id":       <int, 1-indexed> | None,
              "matched_text":     <str> | None,
              "reason":           <str>,
            }
    """
    import jinja2
    from filter_rubric_v4_with_model import parse_json_filter_output

    N = len(samples)
    assert len(adaptive_requirements) == N
    assert len(adaptive_weights) == N
    assert len(general_requirements) == N
    assert len(constraint_requirements) == N

    template = jinja2.Template(dedup_template_str, keep_trailing_newline=True)

    sp_kwargs = dict(judge_cfg["sampling"])
    sp_kwargs["n"] = 1
    enable_thinking = judge_cfg.get("enable_thinking", False)
    thinking_pat = parse_patterns.get("thinking")

    # Identify eligible samples (have adaptive criteria AND at least one reference criterion)
    eligible_indices: list[int] = []
    prompt_texts: list[str] = []
    for i in range(N):
        if not adaptive_requirements[i]:
            continue
        if not general_requirements[i] and not constraint_requirements[i]:
            continue
        eligible_indices.append(i)
        content = template.render(
            instruction=samples[i]["prompt"],
            general_requirements=general_requirements[i],
            constraint_requirements=constraint_requirements[i],
            adaptive_requirements=adaptive_requirements[i],
        )
        prompt_texts.append(
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    # Initialize outputs — non-eligible samples pass through unchanged
    filtered_reqs: list[list[str]] = [list(r) for r in adaptive_requirements]
    filtered_weights: list[list[int]] = [list(w) for w in adaptive_weights]
    parsed_outputs: list[dict | None] = [None] * N
    raw_outputs: list[str] = [""] * N

    # Generate with retries on parse failure
    pending_positions = list(range(len(eligible_indices)))
    for attempt in range(MAX_RETRY + 1):
        if not pending_positions:
            break
        batch = [prompt_texts[p] for p in pending_positions]
        outputs = judge_model.generate(batch, sp_kwargs)

        still_pending: list[int] = []
        for pos_in_batch, p in enumerate(pending_positions):
            raw = outputs[pos_in_batch][0]
            _, resp = split_thinking(raw, thinking_pat)
            i = eligible_indices[p]
            raw_outputs[i] = resp
            parsed = parse_json_filter_output(resp)
            if parsed is not None:
                parsed_outputs[i] = parsed
            else:
                still_pending.append(p)
        pending_positions = still_pending
        if pending_positions and attempt < MAX_RETRY:
            print(f"  Adaptive-dedup retry {attempt + 1}/{MAX_RETRY}: " f"{len(pending_positions)} parse failures")

    # Build dedup_info and apply filtering
    dedup_info: list[dict] = []
    n_filtered_samples = 0
    n_dropped_total = 0
    n_eligible = len(eligible_indices)
    eligible_set = set(eligible_indices)

    for i in range(N):
        adp_reqs = adaptive_requirements[i]
        adp_wts = adaptive_weights[i]
        gen_reqs = general_requirements[i]
        con_reqs = constraint_requirements[i]
        info: dict = {
            "eligible": i in eligible_set,
            "parse_ok": parsed_outputs[i] is not None,
            "raw_output": raw_outputs[i],
            "parsed": parsed_outputs[i],
            "original_adaptive_requirements": list(adp_reqs),
            "original_adaptive_weights": list(adp_wts),
            "kept_ids": list(range(1, len(adp_reqs) + 1)),
            "kept_requirements": list(adp_reqs),
            "kept_weights": list(adp_wts),
            "dropped": [],
        }

        parsed = parsed_outputs[i]
        if parsed is None:
            # Not eligible, or parse failed after retries — keep all adaptive criteria.
            dedup_info.append(info)
            continue

        new_reqs: list[str] = []
        new_weights: list[int] = []
        new_kept_ids: list[int] = []
        dropped: list[dict] = []

        for adp_id in range(1, len(adp_reqs) + 1):
            entry = parsed.get(str(adp_id), {})
            decision = str(entry.get("decision", "keep")).lower()
            adp_text = adp_reqs[adp_id - 1]
            adp_weight = adp_wts[adp_id - 1] if adp_id - 1 < len(adp_wts) else 50

            matched_source = None
            matched_id = None
            matched_text = None
            reason = entry.get("reason", "") or ""

            if decision == "drop":
                dup = entry.get("duplicate_of") or {}
                src = str(dup.get("source", "")).lower() if isinstance(dup, dict) else ""
                raw_id = dup.get("id") if isinstance(dup, dict) else None
                try:
                    ref_id = int(raw_id) if raw_id is not None else None
                except (TypeError, ValueError):
                    ref_id = None

                if src == "general" and ref_id is not None and 1 <= ref_id <= len(gen_reqs):
                    matched_source = "general"
                    matched_id = ref_id
                    matched_text = gen_reqs[ref_id - 1]
                elif src == "constraint" and ref_id is not None and 1 <= ref_id <= len(con_reqs):
                    matched_source = "constraint"
                    matched_id = ref_id
                    matched_text = con_reqs[ref_id - 1]
                else:
                    # Malformed drop — treat as keep (safer default).
                    decision = "keep"

            if decision == "drop":
                dropped.append(
                    {
                        "adaptive_id": adp_id,
                        "adaptive_text": adp_text,
                        "matched_source": matched_source,
                        "matched_id": matched_id,
                        "matched_text": matched_text,
                        "reason": reason,
                    }
                )
            else:
                new_reqs.append(adp_text)
                new_weights.append(adp_weight)
                new_kept_ids.append(adp_id)

        info["kept_ids"] = new_kept_ids
        info["kept_requirements"] = new_reqs
        info["kept_weights"] = new_weights
        info["dropped"] = dropped

        filtered_reqs[i] = new_reqs
        filtered_weights[i] = new_weights
        if dropped:
            n_filtered_samples += 1
            n_dropped_total += len(dropped)

        dedup_info.append(info)

    n_parse_ok = sum(1 for p in parsed_outputs if p is not None)
    print(
        f"  Adaptive-dedup: eligible {n_eligible}/{N}, parsed {n_parse_ok}/{n_eligible}, "
        f"{n_dropped_total} adaptive criteria dropped across {n_filtered_samples} samples"
    )
    return filtered_reqs, filtered_weights, dedup_info
