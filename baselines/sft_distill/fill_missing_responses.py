"""Fill prompts that the original teacher-generation pass dropped.

Loads the existing parquet at ``--output``, compares against the original
dataset (after applying the same overlong-prompt filter), and re-runs only
the missing prompts with a larger ``--max-tokens`` budget. Successful new
responses are appended back into the same parquet.

Usage:
    python baselines/sft_distill/fill_missing_responses.py \\
        --output datasets/sft_distill/wildchecklists_qwen3_8b_teacher.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from generate_teacher_responses import (  # noqa: E402
    extract_user_content,
    generate_with_retry,
    normalize_chat,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BBang3/wildchecklists-with-general")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True, help="Existing parquet to fill in-place")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # Match the SFT trainer's expected response length exactly. Stragglers
    # that don't fit in 4096 will keep getting retried with different seeds
    # until they either land short enough or exhaust ``--max-retry``.
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--max-retry", type=int, default=30)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking (default: enabled, matches RL run)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.exists():
        raise SystemExit(f"[fill] {out_path} does not exist; run generate_teacher_responses.py first")

    existing_df = pd.read_parquet(out_path)
    covered = set(existing_df["prompt"].tolist())
    print(f"[fill] existing rows: {len(existing_df)}, unique prompts: {len(covered)}")

    print(f"[fill] loading dataset {args.dataset}:{args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    chats_all = [normalize_chat(p) for p in ds["prompt"]]
    users_all = [extract_user_content(c) for c in chats_all]

    enable_thinking = not args.no_thinking
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    pending_chats: list[list[dict]] = []
    pending_users: list[str] = []
    overlong = 0
    for chat, user in zip(chats_all, users_all, strict=True):
        if user in covered:
            continue
        templated = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        if len(templated) > args.max_prompt_length:
            overlong += 1
            continue
        pending_chats.append(chat)
        pending_users.append(user)

    print(
        f"[fill] {len(pending_chats)} prompts pending "
        f"(skipped {len(covered)} already covered, {overlong} overlong)"
    )
    if not pending_chats:
        print("[fill] nothing to do")
        return

    print(f"[fill] loading {args.model} (tp={args.tensor_parallel_size})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    base_sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(
        f"[fill] sampling: temp={args.temperature}, top_p={args.top_p}, "
        f"top_k={args.top_k}, max_tokens={args.max_tokens}, "
        f"enable_thinking={enable_thinking}, max_retry={args.max_retry}"
    )

    responses, failed = generate_with_retry(
        llm,
        pending_chats,
        base_sampling=base_sampling,
        base_seed=args.seed,
        enable_thinking=enable_thinking,
        max_retry=args.max_retry,
    )

    new_records = [
        {"prompt": user, "response": resp}
        for user, resp in zip(pending_users, responses, strict=True)
        if resp is not None
    ]

    if failed:
        print(f"[fill] WARNING: {failed} samples still truncated after {args.max_retry} retries")

    if not new_records:
        print("[fill] no new responses to append")
        return

    new_df = pd.DataFrame(new_records)
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["prompt"], keep="first")
    print(
        f"[fill] writing {len(merged)} rows -> {out_path} "
        f"(added {len(new_df)}, still missing {failed})"
    )
    merged.to_parquet(out_path, index=False)
    print("[fill] done")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
