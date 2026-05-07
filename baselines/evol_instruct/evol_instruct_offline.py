"""Offline Evol-Instruct baseline data generation.

Builds a 1-to-1 evolved version of the wildchecklists dataset (no RM-based
selection; pure WizardLM/Evol-Instruct mechanism). Each prompt is taken
through M sequential evolution rounds; per round, one of 5 distilabel-style
methods is sampled and the judge LLM rewrites the prompt. The final
evolved prompt replaces the original; for evolved rows the general rubric
is regenerated from scratch.

Pipeline (per row, batched across all rows each round):
  for round in 1..M:
      Pass 1 (n=1): single sample per row.
      Pass 2 (n=5 retry): for rows that failed Pass 1 (parse / length /
                          unchanged), oversample 5x with the same prompt
                          and method; take the first valid candidate.
      Failures after both passes → chain stays at previous depth.
  Length guard: any candidate whose chat-templated token length exceeds
  ``--max-prompt-length`` is treated as a failure for that round.

  Rubric regen: for rows with depth >= 1, regenerate ``requirements_general``
  and ``weights_general`` via the same judge using the standard rubric
  prompt template. Same 2-pass (n=1 then n=5) retry. If rubric regen still
  fails after retry, the row reverts to the original prompt + original
  rubric (depth set back to 0) so every saved row has a prompt-aligned
  rubric.

Output parquet schema (matches BBang3/wildchecklists-with-general):
  - prompt:                list[{"role": "user", "content": str}]
  - requirements_general:  list[str]
  - weights_general:       list[float]
  - data_source:           original column passed through
  - evol_depth:            int (0..M, 0 means fallback to original)
  - evol_methods:          list[str] (methods that successfully advanced
                           the chain, in order)
  - evol_was_evolved:      bool (depth >= 1 and rubric regen succeeded)
  - rubric_regen_failed:   bool (only true if depth was >=1 but rubric
                           regen failed; row reverted)

Usage
-----
  python baselines/evol_instruct/evol_instruct_offline.py \\
      --output ./datasets/evol_instruct_baseline/wildchecklists_evolved.parquet
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# We reuse the existing rubric parser shipped with VERL so the parsed
# format matches what AdaptiveDataset expects.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl"))
from llm_tutor._parse import (  # noqa: E402
    parse_rubric_output,
    parse_rubric_output_lenient,
)

METHOD_NAMES = [
    "add_constraints",
    "deepen",
    "concretize",
    "increase_reasoning",
    "breadth",
]
# The first 4 are in-depth (use #Rewritten Prompt#:); breadth is in-breadth
# (use #Created Prompt#:).
METHOD_MARKER = {
    "add_constraints": "rewritten",
    "deepen": "rewritten",
    "concretize": "rewritten",
    "increase_reasoning": "rewritten",
    "breadth": "created",
}

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Marker variants Qwen3 actually emits in practice. Colon optional;
# whitespace / markdown formatting (** **) tolerated. The captured form
# matters only for stripping — what we keep is everything after.
REWRITTEN_MARKER_RE = re.compile(
    r"\**\s*#?\s*Rewritten\s*Prompt\s*#?\s*[:：]?\s*\**\s*\n?",
    re.IGNORECASE,
)
CREATED_MARKER_RE = re.compile(
    r"\**\s*#?\s*Created\s*Prompt\s*#?\s*[:：]?\s*\**\s*\n?",
    re.IGNORECASE,
)
GIVEN_MARKER_RE = re.compile(
    r"\**\s*#?\s*(?:The\s+)?Given\s*Prompt\s*#?\s*[:：]?\s*\**",
    re.IGNORECASE,
)


def load_method_templates(prompts_dir: Path) -> dict[str, str]:
    return {m: (prompts_dir / f"{m}.txt").read_text() for m in METHOD_NAMES}


def normalize_chat(prompt) -> list[dict]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


def extract_user_content(chat) -> str:
    if isinstance(chat, str):
        return chat
    for msg in chat:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    raise ValueError(f"No user message in: {chat!r}")


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def parse_evolution(text: str, method: str) -> str | None:
    """Pull the rewritten/created prompt from the model's response.

    Qwen3 tends to omit the canonical ``#Rewritten Prompt#:`` / ``#Created
    Prompt#:`` marker entirely — most outputs are the rewritten prompt
    directly with no prefix. When the marker IS emitted, it's often without
    the colon (e.g. ``#Created Prompt#``) or wrapped in markdown bold.

    Strategy:
      1. Strip the thinking block.
      2. Use the LAST occurrence of a flexible marker regex (colon-optional,
         markdown-tolerant) to split.
      3. Strip any leading marker fragment that wasn't matched cleanly,
         and any trailing ``#Given Prompt#`` (some models echo it back).
      4. Reject empty / too-short results.
    """
    text = strip_thinking(text)
    if not text:
        return None

    marker_re = REWRITTEN_MARKER_RE if METHOD_MARKER[method] == "rewritten" else CREATED_MARKER_RE
    matches = list(marker_re.finditer(text))
    if matches:
        last = matches[-1]
        text = text[last.end() :].strip()
    # Belt-and-suspenders: even after splitting, sometimes a marker fragment
    # remains at the very start (model emits both `#Created Prompt#` AND
    # the rewrite without a clear separator). Strip if found at offset 0.
    text = marker_re.sub("", text, count=1).strip() if marker_re.match(text) else text
    # Drop any trailing echoed `#Given Prompt#` (model sometimes appends it).
    cut = GIVEN_MARKER_RE.search(text)
    if cut is not None:
        text = text[: cut.start()].strip()
    if len(text) < 5:
        return None
    return text


def chat_template_token_len(
    tokenizer, content: str, *, enable_thinking: bool
) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
    )


def render_method_prompt(template: str, prompt: str) -> str:
    return template.replace("{prompt}", prompt)


def call_llm_chat(
    llm: LLM,
    user_messages: list[str],
    *,
    sampling: SamplingParams,
    enable_thinking: bool,
):
    """Wrapper around ``llm.chat`` that wraps each user-string in a chat list."""
    chats = [[{"role": "user", "content": m}] for m in user_messages]
    return llm.chat(
        messages=chats,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        add_generation_prompt=True,
    )


def evolve_one_round(
    llm: LLM,
    *,
    indices: list[int],
    current: list[str],
    methods: list[str],
    templates: dict[str, str],
    tokenizer,
    max_prompt_length: int,
    enable_thinking: bool,
    base_sampling: dict,
    seed_pass1: int,
    seed_pass2: int,
    retry_n: int,
    raw_dump=None,  # optional file handle; if set, write each (round, k, raw_text) as JSONL
    round_no: int | None = None,
) -> tuple[list[bool], list[str], list[dict]]:
    """Run one chain-evolution round across ``indices``.

    Returns:
        success_mask: bool list aligned with ``indices``; True if the row's
                      chain advanced this round.
        new_text:     proposed evolved string for each ``indices[k]``;
                      caller decides whether to accept (only if mask True).
        per_row_log:  per-row debug records (method, pass1_ok, pass2_ok,
                      pass2_attempted, fail_reason).
    """
    if not indices:
        return [], [], []

    user_msgs = [
        render_method_prompt(templates[methods[k]], current[k])
        for k in range(len(indices))
    ]

    pass1_sampling = SamplingParams(
        n=1, seed=seed_pass1, **base_sampling
    )
    pass1_outs = call_llm_chat(
        llm, user_msgs, sampling=pass1_sampling, enable_thinking=enable_thinking
    )

    success_mask = [False] * len(indices)
    new_text = [""] * len(indices)
    per_row_log: list[dict] = []
    failed_pass1: list[int] = []

    for k, out in enumerate(pass1_outs):
        method = methods[k]
        if not out.outputs:
            failed_pass1.append(k)
            per_row_log.append(
                {"method": method, "pass1_ok": False, "pass2_attempted": False, "reason": "empty_output"}
            )
            continue
        completion = out.outputs[0]
        if raw_dump is not None:
            raw_dump.write(json.dumps({
                "round": round_no, "row": k, "pass": 1, "method": method,
                "finish_reason": completion.finish_reason,
                "raw_text": completion.text,
            }, ensure_ascii=False) + "\n")
        ok, text, reason = _validate_candidate(
            completion=completion,
            method=method,
            previous=current[k],
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            enable_thinking=enable_thinking,
        )
        if ok:
            success_mask[k] = True
            new_text[k] = text
            per_row_log.append({"method": method, "pass1_ok": True, "pass2_attempted": False, "reason": None})
        else:
            failed_pass1.append(k)
            per_row_log.append(
                {"method": method, "pass1_ok": False, "pass2_attempted": False, "reason": reason}
            )

    # Pass 2: oversample n=retry_n only for the failures.
    if failed_pass1:
        pass2_sampling = SamplingParams(
            n=retry_n, seed=seed_pass2, **base_sampling
        )
        pass2_msgs = [user_msgs[k] for k in failed_pass1]
        pass2_outs = call_llm_chat(
            llm, pass2_msgs, sampling=pass2_sampling, enable_thinking=enable_thinking
        )
        for j, out in enumerate(pass2_outs):
            k = failed_pass1[j]
            method = methods[k]
            row_log = per_row_log[k]
            row_log["pass2_attempted"] = True
            rescued = False
            last_reason = row_log.get("reason") or "unknown"
            for completion in out.outputs:
                ok, text, reason = _validate_candidate(
                    completion=completion,
                    method=method,
                    previous=current[k],
                    tokenizer=tokenizer,
                    max_prompt_length=max_prompt_length,
                    enable_thinking=enable_thinking,
                )
                if ok:
                    success_mask[k] = True
                    new_text[k] = text
                    row_log["pass2_ok"] = True
                    row_log["reason"] = None
                    rescued = True
                    break
                last_reason = reason
            if not rescued:
                row_log["pass2_ok"] = False
                row_log["reason"] = last_reason

    return success_mask, new_text, per_row_log


def _validate_candidate(
    *,
    completion,
    method: str,
    previous: str,
    tokenizer,
    max_prompt_length: int,
    enable_thinking: bool,
) -> tuple[bool, str, str | None]:
    """Return (ok, parsed_text, fail_reason)."""
    if completion.finish_reason == "length":
        # The thinking block ran the model out of tokens; whatever came
        # before is partial — reject so we retry with a different seed.
        return False, "", "length_truncated"
    parsed = parse_evolution(completion.text, method)
    if parsed is None:
        return False, "", "parse_fail"
    if parsed == previous:
        return False, "", "no_change"
    n_tok = chat_template_token_len(tokenizer, parsed, enable_thinking=enable_thinking)
    if n_tok > max_prompt_length:
        return False, "", "length_exceeded"
    return True, parsed, None


def regen_general_rubric(
    llm: LLM,
    rubric_template: str,
    prompts: list[str],
    *,
    seed_pass1: int,
    seed_pass2: int,
    retry_n: int,
    base_sampling: dict,
    enable_thinking: bool,
) -> tuple[list[list[str] | None], list[list[float] | None], int, int]:
    """Regenerate (criteria, weights) for each prompt. Returns tuple of
    (criteria_list, weights_list, pass1_ok_count, pass2_ok_count). For rows
    where both passes fail, the entry is (None, None) — caller handles
    fallback."""
    if not prompts:
        return [], [], 0, 0

    user_msgs = [rubric_template.replace("{instruction}", p) for p in prompts]
    pass1_sampling = SamplingParams(n=1, seed=seed_pass1, **base_sampling)
    outs = call_llm_chat(
        llm, user_msgs, sampling=pass1_sampling, enable_thinking=enable_thinking
    )

    criteria_out: list[list[str] | None] = [None] * len(prompts)
    weights_out: list[list[float] | None] = [None] * len(prompts)
    failed: list[int] = []
    pass1_ok = 0

    for i, out in enumerate(outs):
        if not out.outputs or out.outputs[0].finish_reason == "length":
            failed.append(i)
            continue
        text = strip_thinking(out.outputs[0].text)
        _, pairs = parse_rubric_output(text)
        if not pairs:
            _, pairs = parse_rubric_output_lenient(text)
        if not pairs:
            failed.append(i)
            continue
        criteria_out[i] = [p["rubric"] for p in pairs]
        weights_out[i] = [float(p["importance"]) for p in pairs]
        pass1_ok += 1

    pass2_ok = 0
    if failed:
        pass2_sampling = SamplingParams(n=retry_n, seed=seed_pass2, **base_sampling)
        pass2_msgs = [user_msgs[i] for i in failed]
        outs2 = call_llm_chat(
            llm, pass2_msgs, sampling=pass2_sampling, enable_thinking=enable_thinking
        )
        for j, out in enumerate(outs2):
            i = failed[j]
            for completion in out.outputs:
                if completion.finish_reason == "length":
                    continue
                text = strip_thinking(completion.text)
                _, pairs = parse_rubric_output(text)
                if not pairs:
                    _, pairs = parse_rubric_output_lenient(text)
                if pairs:
                    criteria_out[i] = [p["rubric"] for p in pairs]
                    weights_out[i] = [float(p["importance"]) for p in pairs]
                    pass2_ok += 1
                    break

    return criteria_out, weights_out, pass1_ok, pass2_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BBang3/wildchecklists-with-general")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--prompts-dir", default=None,
                        help="Override path to evol-instruct prompt templates")
    parser.add_argument(
        "--rubric-template",
        default="data_generation/prompts/base_rubric_generation.txt",
        help="General-rubric prompt template (uses {instruction} placeholder)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=8192,
        help="Match data.max_prompt_length in the RL config; evolved prompts "
        "exceeding this are treated as failures.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--turns", type=int, default=4, help="M, number of evolution rounds")
    parser.add_argument("--retry-n", type=int, default=5,
                        help="Pass-2 oversample size when Pass-1 fails")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking (default: enabled, matches the RL judge)",
    )
    parser.add_argument(
        "--report-jsonl",
        default=None,
        help="If set, dump per-row chain audit log to this path",
    )
    parser.add_argument(
        "--raw-jsonl",
        default=None,
        help="If set, dump every Pass-1 raw model output to this JSONL "
        "(one record per row per round). Useful for debugging marker parsing.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else (repo_root / "baselines/evol_instruct/prompts")
    rubric_template_path = repo_root / args.rubric_template
    if not rubric_template_path.exists():
        rubric_template_path = Path(args.rubric_template)
    rubric_template = rubric_template_path.read_text()
    method_templates = load_method_templates(prompts_dir)
    enable_thinking = not args.no_thinking

    print(f"[evol] loading {args.dataset}:{args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples > 0 and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"[evol] dataset size: {len(ds)}")

    raw_chats = [normalize_chat(p) for p in ds["prompt"]]
    raw_users = [extract_user_content(c) for c in raw_chats]
    raw_reqs = list(ds["requirements_general"]) if "requirements_general" in ds.column_names else [None] * len(ds)
    raw_weights = list(ds["weights_general"]) if "weights_general" in ds.column_names else [None] * len(ds)
    raw_source = list(ds["data_source"]) if "data_source" in ds.column_names else [None] * len(ds)

    print(f"[evol] tokenizing for length filter (>{args.max_prompt_length} tokens)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    keep_mask: list[bool] = []
    for user in raw_users:
        n_tok = chat_template_token_len(tokenizer, user, enable_thinking=enable_thinking)
        keep_mask.append(n_tok <= args.max_prompt_length)
    n_dropped = keep_mask.count(False)
    print(f"[evol] dropping {n_dropped} overlong prompts; {keep_mask.count(True)} remain")

    # Index map: position in filtered list → position in original.
    kept_idx = [i for i, k in enumerate(keep_mask) if k]
    N = len(kept_idx)

    current = [raw_users[i] for i in kept_idx]
    last_good_depth = [0] * N
    method_history: list[list[str]] = [[] for _ in range(N)]

    print(f"[evol] loading {args.model} (tp={args.tensor_parallel_size})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    base_sampling = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    rng = np.random.default_rng()
    round_stats: list[dict] = []

    raw_dump_handle = None
    if args.raw_jsonl:
        raw_dump_path = Path(args.raw_jsonl)
        raw_dump_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dump_handle = open(raw_dump_path, "w")
        print(f"[evol] raw model outputs dumping to {raw_dump_path}")

    for round_idx in range(args.turns):
        round_no = round_idx + 1
        # All N rows participate every round; chain just stays put on failure.
        methods_this_round = [METHOD_NAMES[rng.integers(0, len(METHOD_NAMES))] for _ in range(N)]
        print(
            f"[evol] round {round_no}/{args.turns}: "
            f"chatting on {N} prompts (pass1 n=1)"
        )
        success_mask, new_text, per_row_log = evolve_one_round(
            llm,
            indices=list(range(N)),
            current=current,
            methods=methods_this_round,
            templates=method_templates,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            enable_thinking=enable_thinking,
            base_sampling=base_sampling,
            seed_pass1=1000 + round_no,
            seed_pass2=2000 + round_no,
            retry_n=args.retry_n,
            raw_dump=raw_dump_handle,
            round_no=round_no,
        )

        pass1_ok = sum(1 for log in per_row_log if log.get("pass1_ok"))
        pass2_attempted = sum(1 for log in per_row_log if log.get("pass2_attempted"))
        pass2_ok = sum(1 for log in per_row_log if log.get("pass2_ok"))
        final_fail = pass2_attempted - pass2_ok
        round_stats.append(
            {"round": round_no, "pass1_ok": pass1_ok, "pass2_attempted": pass2_attempted,
             "pass2_ok": pass2_ok, "final_fail": final_fail}
        )
        print(
            f"[evol] round {round_no}: pass1_ok={pass1_ok} "
            f"pass2_attempted={pass2_attempted} pass2_ok={pass2_ok} "
            f"final_fail={final_fail}"
        )

        for k in range(N):
            if success_mask[k]:
                current[k] = new_text[k]
                last_good_depth[k] = round_no
                method_history[k].append(methods_this_round[k])

    n_evolved = sum(1 for d in last_good_depth if d >= 1)
    print(f"[evol] chain done; {n_evolved}/{N} rows reached depth >= 1")

    # Rubric regen for evolved rows.
    print(f"[evol] regenerating general rubric for {n_evolved} evolved rows")
    evolved_local_idx = [k for k in range(N) if last_good_depth[k] >= 1]
    evolved_prompts = [current[k] for k in evolved_local_idx]
    new_reqs, new_weights, rub_pass1_ok, rub_pass2_ok = regen_general_rubric(
        llm,
        rubric_template,
        evolved_prompts,
        seed_pass1=9001,
        seed_pass2=9002,
        retry_n=args.retry_n,
        base_sampling=base_sampling,
        enable_thinking=enable_thinking,
    )
    rubric_failed_local: set[int] = set()
    for j, k in enumerate(evolved_local_idx):
        if new_reqs[j] is None:
            rubric_failed_local.add(k)

    print(
        f"[evol] rubric regen: pass1_ok={rub_pass1_ok} pass2_ok={rub_pass2_ok} "
        f"final_fail={len(rubric_failed_local)} (these revert to original)"
    )

    # Build output rows.
    out_rows = []
    rev_evolved_idx = {k: j for j, k in enumerate(evolved_local_idx)}
    for k in range(N):
        orig_i = kept_idx[k]
        was_evolved = last_good_depth[k] >= 1 and k not in rubric_failed_local
        rubric_regen_failed = k in rubric_failed_local
        if was_evolved:
            j = rev_evolved_idx[k]
            user_text = current[k]
            requirements = list(new_reqs[j])
            weights = list(new_weights[j])
            depth = last_good_depth[k]
            methods = list(method_history[k])
        else:
            user_text = raw_users[orig_i]
            requirements = list(raw_reqs[orig_i]) if raw_reqs[orig_i] is not None else []
            weights = list(raw_weights[orig_i]) if raw_weights[orig_i] is not None else []
            depth = 0
            methods = []

        out_rows.append({
            "prompt": [{"role": "user", "content": user_text}],
            "requirements_general": requirements,
            "weights_general": weights,
            "data_source": raw_source[orig_i],
            "evol_depth": depth,
            "evol_methods": methods,
            "evol_was_evolved": was_evolved,
            "rubric_regen_failed": rubric_regen_failed,
        })

    df = pd.DataFrame(out_rows)
    print(f"[evol] writing {len(df)} rows -> {out_path}")
    df.to_parquet(out_path, index=False)

    # Fallback / depth distribution report.
    depth_counter: Counter[int] = Counter(int(d) for d in df["evol_depth"])
    method_counter: Counter[str] = Counter()
    for methods in df["evol_methods"]:
        method_counter.update(methods)

    print("\n=== Evol-Instruct fallback report ===")
    print(f"Total rows kept (after overlong filter): {N}")
    print(f"Final depth distribution:")
    for d in range(args.turns + 1):
        cnt = depth_counter.get(d, 0)
        pct = 100.0 * cnt / max(1, N)
        tag = " (fallback to original)" if d == 0 else ""
        print(f"  depth {d}: {cnt} ({pct:.1f}%){tag}")
    print("Per-round breakdown:")
    for s in round_stats:
        print(
            f"  round {s['round']}: pass1_ok={s['pass1_ok']} "
            f"pass2_attempted={s['pass2_attempted']} pass2_ok={s['pass2_ok']} "
            f"final_fail={s['final_fail']}"
        )
    rescue_total = sum(s["pass2_ok"] for s in round_stats)
    rescue_attempt = sum(s["pass2_attempted"] for s in round_stats)
    rescue_rate = (rescue_total / rescue_attempt) if rescue_attempt else 0.0
    print(f"Retry rescue rate: {rescue_total}/{rescue_attempt} = {rescue_rate:.1%}")
    print("Successful method usage (across all rounds):")
    for m in METHOD_NAMES:
        print(f"  {m}: {method_counter.get(m, 0)}")
    print(f"Rubric regen: {rub_pass1_ok + rub_pass2_ok}/{n_evolved} succeeded "
          f"({len(rubric_failed_local)} reverted to original)")
    print("===")

    if raw_dump_handle is not None:
        raw_dump_handle.close()

    if args.report_jsonl:
        report_path = Path(args.report_jsonl)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps({
                    "prompt": row["prompt"][0]["content"][:200],
                    "evol_depth": row["evol_depth"],
                    "evol_methods": row["evol_methods"],
                    "evol_was_evolved": row["evol_was_evolved"],
                    "rubric_regen_failed": row["rubric_regen_failed"],
                }) + "\n")
        print(f"[evol] per-row audit written to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
