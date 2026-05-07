from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
from typing import Any

import aiohttp

from templates.utils import load_template


@functools.lru_cache(maxsize=16)
def _load_template_cached(template_name: str) -> str:
    return load_template(template_name)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LEADING_NUMBER_PATTERN = re.compile(r"^\s*\d+\)\s*")
_IMPORTANCE_SPLIT_PATTERN = re.compile(
    r"(.*?)\(\s*importance\s*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_wildchecklist_requirements(requirements_text: str) -> list[tuple[str, float]]:
    if not _IMPORTANCE_SPLIT_PATTERN.search(requirements_text):
        raise ValueError(f"No (importance: X/Y) tag found in requirements text: {requirements_text!r}")

    requirements: list[tuple[str, float]] = []
    for match in _IMPORTANCE_SPLIT_PATTERN.finditer(requirements_text):
        raw_text = match.group(1)
        numerator = float(match.group(2))
        denominator = float(match.group(3))
        weight = (numerator / denominator) if denominator > 0.0 else None
        if weight is None or not (0.0 <= weight <= 1.0):
            raise ValueError(f"Invalid importance weight: {weight} (numerator={numerator}, denominator={denominator})")

        lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if not lines:
                line = _LEADING_NUMBER_PATTERN.sub("", line).strip()
            if line:
                lines.append(line)

        text = "\n".join(lines).strip()
        if text:
            requirements.append((text, weight))

    return requirements


def _remove_thinking_from_completion(completion: str) -> str:
    if "<think>" in completion and "</think>" in completion:
        return completion.split("</think>")[-1].strip()
    return completion.strip()




def _extract_score(judge_completion: str) -> float | None:
    score_tag_pattern = re.compile(r"<score>\s*([-+]?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE | re.DOTALL)
    score_tag_brace_pattern = re.compile(
        r"<score>\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*</score>", re.IGNORECASE | re.DOTALL
    )
    base_number_pattern = re.compile(r"(?:^|[\n\r])\s*([-+]?\d+(?:\.\d+)?)\s*(?:[\n\r]|$)")

    matches = score_tag_pattern.findall(judge_completion)
    if matches:
        return float(matches[-1])

    matches = score_tag_brace_pattern.findall(judge_completion)
    if matches:
        return float(matches[-1])

    matches = base_number_pattern.findall(judge_completion)
    if matches:
        return float(matches[-1])

    return None


def _clamp_reward_score(
    score: float,
    parse_fail_score: float,
    min_score: float | None,
    max_score: float | None,
) -> float:
    if not math.isfinite(score):
        return float(parse_fail_score)
    if min_score is not None:
        score = max(score, float(min_score))
    if max_score is not None:
        score = min(score, float(max_score))
    return float(score)


async def _generate_single_chat_completion(
    prompt: str,
    request_url: str,
    payload_base: dict[str, Any],
    session: aiohttp.ClientSession,
) -> str:
    payload = {**payload_base, "messages": [{"role": "user", "content": prompt}]}
    async with session.post(request_url, json=payload) as response:
        response.raise_for_status()
        output = await response.json()
    choices = output.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Empty response from /v1/chat/completions")
    message = choices[0].get("message", {})
    return message.get("content", "") or ""


async def _generate_judge_completions_batch(
    prompts: list[str],
    reward_router_address: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    session: aiohttp.ClientSession,
    judge_model_name: str | None = None,
    enable_thinking: bool = False,
) -> list[str]:
    if not prompts:
        return []

    payload_base: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if top_k is not None and int(top_k) >= 0:
        payload_base["top_k"] = int(top_k)
    if judge_model_name:
        payload_base["model"] = judge_model_name
    if enable_thinking:
        payload_base["chat_template_kwargs"] = {"enable_thinking": True}

    request_url = f"http://{reward_router_address}/v1/chat/completions"

    tasks = [
        _generate_single_chat_completion(
            prompt=prompt,
            request_url=request_url,
            payload_base=payload_base,
            session=session,
        )
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks)


async def _run_judge_completions(
    prompts: list[str],
    *,
    reward_router_address: str | None,
    judge_model_name: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    enable_thinking: bool,
) -> list[str]:
    """Dispatch judge prompts to the in-cluster vLLM router (one POST per prompt)."""
    if not prompts:
        return []
    if not reward_router_address:
        raise ValueError("judge dispatch requires reward_router_address (in-cluster vLLM router)")
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=connector) as session:
        return await _generate_judge_completions_batch(
            prompts=prompts,
            reward_router_address=reward_router_address,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            session=session,
            judge_model_name=judge_model_name,
            enable_thinking=enable_thinking,
        )


async def compute_score_rubric_list(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    reward_router_address: str | None = None,
    template_name: str = "judge.txt",
    requirements_key: str = "requirements",
    weights_key: str = "weights",
    instruction_key: str = "prompt",
    score_aggregation: str | None = "weighted_mean",
    parse_fail_score: float = 0.0,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    judge_model_name: str | None = None,
    enable_thinking: bool = False,
    enforce_score_bounds: bool = True,
    min_score: float | None = -1.0,
    max_score: float | None = 100.0,
    reward_model_tokenizer=None,
) -> dict[str, Any]:
    del data_source, ground_truth, reward_model_tokenizer
    instruction = extra_info.get(instruction_key)
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = extra_info.get("prompt_str")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"Missing instruction text: instruction_key={instruction_key!r}, "
            f"extra_info_keys={sorted(extra_info.keys())}"
        )

    raw_requirements = extra_info.get(requirements_key) or []
    if isinstance(raw_requirements, str):
        try:
            requirement_texts = json.loads(raw_requirements)
        except json.JSONDecodeError as e:
            raise ValueError(f"requirements key {requirements_key!r} is a string but not valid JSON: {e}") from e
    else:
        requirement_texts = raw_requirements
    if not isinstance(requirement_texts, list) or not requirement_texts:
        raise ValueError(
            f"requirements must be a non-empty list; got {type(requirement_texts).__name__!r} "
            f"from key {requirements_key!r}"
        )

    raw_weights = extra_info.get(weights_key)
    if score_aggregation == "weighted_mean":
        if isinstance(raw_weights, str):
            try:
                raw_weights = json.loads(raw_weights)
            except json.JSONDecodeError as e:
                raise ValueError(f"weights key {weights_key!r} is a string but not valid JSON: {e}") from e
        if not isinstance(raw_weights, list) or not raw_weights:
            raise ValueError(
                f"weighted_mean requires a non-empty list of weights from key {weights_key!r}; "
                f"got {type(raw_weights).__name__!r}"
            )
        importance_weights = [float(w) for w in raw_weights]
        if len(importance_weights) != len(requirement_texts):
            raise ValueError(
                f"weights length ({len(importance_weights)}) != requirements length ({len(requirement_texts)})"
            )
    else:
        importance_weights = [1.0] * len(requirement_texts)

    if enable_thinking:
        if "<think>" in solution_str:
            if "</think>" in solution_str:
                solution_str = solution_str.split("</think>")[-1].strip()
            else:
                solution_str = "<invalid_response>"
            logger.debug("Thinking block removed from response.")
        # else:
        #     logger.warning(
        #         f"Thinking is enabled, but <think> is not found in the response: {solution_str if len(solution_str) < 100 else solution_str[:100] + '...'}"
        #     )

    template = _load_template_cached(template_name)
    requirement_scores: list[float] = []
    requirement_weights: list[float] = []
    parse_success_flags: list[bool] = []

    judge_prompts = [
        template.format(
            instruction=instruction,
            response=solution_str,
            requirement=req_text,
        )
        for req_text in requirement_texts
    ]

    judge_outputs = await _run_judge_completions(
        prompts=judge_prompts,
        reward_router_address=reward_router_address,
        judge_model_name=judge_model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )

    for importance_weight, judge_completion in zip(importance_weights, judge_outputs, strict=True):
        if enable_thinking:
            judge_completion = _remove_thinking_from_completion(judge_completion)
        parsed_score = _extract_score(judge_completion)
        parse_success = parsed_score is not None
        score = float(parsed_score) if parse_success else float(parse_fail_score)
        if enforce_score_bounds:
            score = _clamp_reward_score(
                score=score,
                parse_fail_score=parse_fail_score,
                min_score=min_score,
                max_score=max_score,
            )

        parse_success_flags.append(bool(parse_success))
        requirement_scores.append(float(score))
        if score_aggregation == "weighted_mean":
            requirement_weights.append(float(importance_weight))
        else:
            requirement_weights.append(1.0)

    weighted_scores = [score * weight for score, weight in zip(requirement_scores, requirement_weights, strict=True)]

    if score_aggregation == "mean":
        total_score = float(sum(weighted_scores) / max(len(weighted_scores), 1))
    elif score_aggregation == "weighted_mean":
        total_weight = float(sum(requirement_weights))
        total_score = float(sum(weighted_scores) / total_weight) if total_weight > 0 else 0.0
    else:
        raise ValueError(f"Unsupported score_aggregation: {score_aggregation}")

    parse_success_count = sum(1 for flag in parse_success_flags if flag)
    parse_success_rate = parse_success_count / max(len(parse_success_flags), 1)
    reward_score = total_score / 100.0

    return {
        "score": reward_score,
        "rubric_total_score": total_score,
        "rubric_num_requirements": len(requirement_scores),
        "rubric_parse_success_bool": json.dumps(parse_success_flags, ensure_ascii=False),
        "rubric_parse_success_count": parse_success_count,
        "rubric_parse_success_rate": parse_success_rate,
        "rubric_score_aggregation": score_aggregation,
        "rubric_requirements": requirement_texts,
        "rubric_judgement_scores": json.dumps(requirement_scores, ensure_ascii=False),
        "rubric_judgement_completions": judge_outputs,
    }

