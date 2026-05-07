"""Offline teacher-response generation for the SFT distillation baseline.

Loads the wildchecklists prompts used by the RL run, runs Qwen3-8B (the same
model used as both reward model and rubric generator in the RL pipeline) on
each prompt once, and writes a parquet file with two columns ``prompt`` (the
user message as a single string) and ``response`` (the teacher's full output,
including the ``<think>...</think>`` block).

Sampling matches the RL ``data_generator`` defaults so that the teacher's
output distribution is the same one the policy sees as a reward signal during
PPO. This keeps the comparison "RL on reward signal vs SFT on the underlying
generation" clean.

Truncation handling
-------------------
A response is "good" only if vLLM reports ``finish_reason == "stop"`` AND the
text contains a closing ``</think>`` tag (when thinking is enabled). Anything
that hit ``max_tokens`` mid-thought would land in the SFT target as a
half-finished response, which corrupts the student. We retry such samples up
to ``--max-retry`` times with a different seed; samples that still fail after
all retries are dropped from the output (count is reported).

Usage
-----
    python baselines/sft_distill/generate_teacher_responses.py \\
        --output ./datasets/sft_distill/wildchecklists_qwen3_8b_teacher.parquet

Smoke test on 32 rows:
    python baselines/sft_distill/generate_teacher_responses.py \\
        --output /tmp/teacher_smoke.parquet --max-samples 32
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_user_content(chat) -> str:
    """Return the user-role content from a chat list, or the string itself."""
    if isinstance(chat, str):
        return chat
    for msg in chat:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "user":
            return msg["content"]
    raise ValueError(f"No user message found in prompt: {chat!r}")


def normalize_chat(prompt) -> list[dict]:
    """Coerce the dataset's ``prompt`` cell into a list[{role, content}]."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    # HF datasets / pyarrow may yield numpy arrays of dicts; force to list.
    return [dict(m) for m in prompt]


def is_good_response(out, *, enable_thinking: bool) -> bool:
    """A response is acceptable if it stopped naturally and (when thinking is
    enabled) closed its ``<think>`` block. Truncation due to the thinking
    budget — the most common failure mode for Qwen3 at max_tokens=4k — is
    detected by either ``finish_reason == "length"`` or a missing
    ``</think>`` tag (model still inside the thinking block when cut off).
    """
    if not out.outputs:
        return False
    completion = out.outputs[0]
    if completion.finish_reason != "stop":
        return False
    if enable_thinking and "</think>" not in completion.text:
        return False
    if not completion.text.strip():
        return False
    return True


def generate_with_retry(
    llm: LLM,
    chats: list[list[dict]],
    *,
    base_sampling: SamplingParams,
    base_seed: int,
    enable_thinking: bool,
    max_retry: int,
) -> tuple[list[str | None], int]:
    """Generate a response per chat, retrying truncated samples up to
    ``max_retry`` times. Returns a list of response strings (None for samples
    that failed all attempts) and the count of final failures."""

    n = len(chats)
    responses: list[str | None] = [None] * n
    pending: list[int] = list(range(n))

    for attempt in range(max_retry + 1):
        if not pending:
            break
        sampling = base_sampling.clone()
        sampling.seed = base_seed + attempt  # vary seed across attempts
        print(
            f"[teacher-gen] attempt {attempt + 1}/{max_retry + 1}: "
            f"generating {len(pending)} samples (seed={sampling.seed})"
        )
        sub_chats = [chats[i] for i in pending]
        outputs = llm.chat(
            messages=sub_chats,
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            add_generation_prompt=True,
        )

        next_pending: list[int] = []
        for idx, out in zip(pending, outputs, strict=True):
            if is_good_response(out, enable_thinking=enable_thinking):
                responses[idx] = out.outputs[0].text
            else:
                next_pending.append(idx)
        rescued = len(pending) - len(next_pending)
        print(
            f"[teacher-gen] attempt {attempt + 1}: "
            f"{rescued} settled, {len(next_pending)} still truncated"
        )
        pending = next_pending

    failed = len(pending)
    return responses, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BBang3/wildchecklists-with-general")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=8192,
        help="Drop prompts whose chat-templated token count exceeds this. "
        "Matches RL data.max_prompt_length / filter_overlong_prompts=true.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 = all")
    parser.add_argument(
        "--max-retry",
        type=int,
        default=3,
        help="Retry budget for samples that hit max_tokens (default: 3)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking (default: enabled, matches RL run)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[teacher-gen] loading dataset {args.dataset}:{args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples > 0 and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"[teacher-gen] dataset size: {len(ds)}")

    chats = [normalize_chat(p) for p in ds["prompt"]]
    user_contents = [extract_user_content(c) for c in chats]

    enable_thinking = not args.no_thinking

    # Pre-filter overlong prompts so vLLM never sees a prompt that exceeds
    # max_model_len. Match the RL pipeline's filter_overlong_prompts behaviour:
    # tokens are counted *after* applying the chat template (with the same
    # enable_thinking flag), and the cutoff is data.max_prompt_length.
    print(
        f"[teacher-gen] tokenizing {len(chats)} prompts to filter > "
        f"{args.max_prompt_length} tokens (chat-templated)"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    keep_chats: list[list[dict]] = []
    keep_users: list[str] = []
    dropped = 0
    for chat, user in zip(chats, user_contents, strict=True):
        templated = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        if len(templated) <= args.max_prompt_length:
            keep_chats.append(chat)
            keep_users.append(user)
        else:
            dropped += 1
    if dropped:
        print(f"[teacher-gen] filtered {dropped} overlong prompts; {len(keep_chats)} remain")
    chats = keep_chats
    user_contents = keep_users

    print(f"[teacher-gen] loading {args.model} (tp={args.tensor_parallel_size})")
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
        f"[teacher-gen] sampling: temp={args.temperature}, top_p={args.top_p}, "
        f"top_k={args.top_k}, max_tokens={args.max_tokens}, "
        f"enable_thinking={enable_thinking}, max_retry={args.max_retry}"
    )

    responses, failed = generate_with_retry(
        llm,
        chats,
        base_sampling=base_sampling,
        base_seed=args.seed,
        enable_thinking=enable_thinking,
        max_retry=args.max_retry,
    )

    records = [
        {"prompt": user, "response": resp}
        for user, resp in zip(user_contents, responses, strict=True)
        if resp is not None
    ]

    if failed:
        print(
            f"[teacher-gen] WARNING: dropped {failed} samples that remained "
            f"truncated after {args.max_retry} retries"
        )

    df = pd.DataFrame(records)
    print(f"[teacher-gen] writing {len(df)} rows -> {out_path}")
    df.to_parquet(out_path, index=False)
    print(f"[teacher-gen] done (kept {len(df)} / {len(chats)})")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
