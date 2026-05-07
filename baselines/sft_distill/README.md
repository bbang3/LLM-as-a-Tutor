# SFT Distillation Baseline (Qwen3-8B → Qwen3-1.7B)

A controlled baseline against the rubric-RL run in
`configs/llm_tutor_base.yaml`. Same prompt set, same student model,
same eval. The only thing that differs is the learning signal: instead of
PPO + reward-model judging, we have the 8B teacher generate one response per
prompt and SFT the 1.7B student on the resulting `(prompt, response)` pairs.

## Why this baseline

The RL run uses Qwen3-8B as both the rubric generator and the reward judge.
A reviewer asking "is the gain from RL or just from injecting 8B-quality
signal into the 1.7B?" needs a fair distillation control. This baseline
answers that — same data, same epoch budget, standard SFT hyperparameters.

## Pipeline

```
HF dataset (BBang3/wildchecklists-with-general)
         │
         │  generate_teacher_responses.py    Qwen3-8B, n=1, temp=0.6
         ▼
    parquet (prompt: str, response: str)
         │
         │  fill_missing_responses.py        retry truncated samples
         ▼
    parquet (≈3867 rows after drops)
         │
         │  run_sft_distill.sh               Qwen3-1.7B, prompt-mask SFT
         ▼
    checkpoints/.../sft_distill_v1.0_base/
```

## What gets dropped

1. **Overlong prompts** — anything whose chat-templated token length exceeds
   `max_prompt_length=8192`. Mirrors `data.filter_overlong_prompts=true`
   from the RL config so both runs see the same prompt set. Typically 6
   prompts on the wildchecklists set.
2. **Truncated teacher responses** — anything where the 8B teacher hit
   `max_tokens=4096` mid-thinking (no closing `</think>`) and never
   recovered across retry attempts. The student's response budget is
   also 4096 in the RL run, so a teacher response that doesn't fit is
   one we don't want as an SFT target either. Typically ≈100–150 prompts.

After the standard run on the 4000-row dataset: 4000 → 3994 (overlong
filter) → 3867 (truncation drop). Drop rate ≈3.3%; the corresponding
prompt list is logged so the same set can be excluded from RL ablations
if a strict ablation is needed.

## Hyperparameters and rationale

| Knob | Value | Why |
|---|---|---|
| `train_batch_size` | 16 | At 4k rows × 3 epochs that's ~750 grad steps, same order as the RL run's PPO updates. |
| `lr` | 1e-5 | Standard SFT; one decade above the RL `actor.optim.lr=5e-6`. |
| `lr_scheduler` | cosine | SFT-typical decay over a fixed dataset (RL uses constant + warmup; not appropriate when the data isn't moving). |
| `lr_warmup_steps_ratio` | 0.05 | ≈37 step warmup on 750 — close to the RL run's absolute 25-step warmup. |
| `weight_decay` | 0.01 | AdamW default. |
| `betas` | [0.9, 0.95] | LLM-standard (lower β2 than torch default 0.999), matches RL config. |
| `clip_grad` | 1.0 | Cheap insurance against bf16 outlier spikes. |
| `total_epochs` | 3 | Matches RL `total_epochs`. |
| `max_length` | 12288 | 8k prompt + 4k response. Same budgets as RL. |
| Loss mask | response only | Standard prompt-mask SFT. |
| `apply_chat_template_kwargs.enable_thinking` | true | Same as RL. Teacher's `<think>...</think>` is part of the SFT target so the student learns to think too. |

The full config is in `sft_distill_v1.0_base.yaml`.

## How to run

The whole pipeline assumes the repo-local venv at `.venv/` (the system
Python doesn't have `verl` installed). Run all commands from the repo
root.

### GPU configuration

Pick a `CUDA_VISIBLE_DEVICES` and `N_GPUS` (= number of devices in that
list) that fit your environment. The examples below use `0,1` and
`N_GPUS=2`; replace with whatever you actually have free
(e.g. `CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4`).

For teacher generation (steps 1–2), `--tensor-parallel-size` should equal
`N_GPUS`. Qwen3-8B fits comfortably with TP=1 on a single 80 GB device,
or with TP=2 across two smaller cards; larger TP also works but isn't
needed.

For SFT (step 3), the launcher reads the GPU count from `NPROC`; pass
the matching `trainer.n_gpus_per_node` Hydra override so VERL builds the
right device mesh. The student is 1.7B so anything from 1 to 8 GPUs is
fine; with the default `train_batch_size=16`, `N_GPUS × micro_batch_size`
must divide 16 (default `micro_batch_size_per_gpu=2` works for 1, 2, 4,
or 8 GPUs).

### 1. Generate teacher responses

```bash
mkdir -p datasets/sft_distill baselines/sft_distill/logs

CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
.venv/bin/python -u baselines/sft_distill/generate_teacher_responses.py \
    --output datasets/sft_distill/wildchecklists_qwen3_8b_teacher.parquet \
    --tensor-parallel-size ${N_GPUS} \
    --max-retry 5 \
    > baselines/sft_distill/logs/teacher_gen.log 2>&1 &
```

Drop count, retry rounds, and the final saved-row total are all in the log.

### 2. Fill in truncated stragglers

The first pass uses `max_retry=5` for speed; whatever didn't settle gets
retried with a fresh seed schedule. Set `--max-retry` to whatever you're
willing to wait for — we use 10 by default.

```bash
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
.venv/bin/python -u baselines/sft_distill/fill_missing_responses.py \
    --output datasets/sft_distill/wildchecklists_qwen3_8b_teacher.parquet \
    --tensor-parallel-size ${N_GPUS} \
    --max-retry 10 \
    > baselines/sft_distill/logs/teacher_fill.log 2>&1 &
```

The script reads the existing parquet, computes which prompts are still
missing (using the same overlong filter as step 1 to skip filtered-out
prompts), generates only those, and **appends + dedupes** in place.

### 3. Train the student

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
nohup bash baselines/sft_distill/run_sft_distill.sh \
    trainer.n_gpus_per_node=${NPROC} \
    > baselines/sft_distill/logs/sft.log 2>&1 &
```

The config defaults to `n_gpus_per_node=4`; the Hydra override above keeps
it consistent with `NPROC` when you're on a different count. Checkpoints
land in `checkpoints/prompting-llm-verl/sft_distill_v1.0_base/`.

### 4. Evaluate

Run benchmark evaluation against the SFT checkpoint directory using your
preferred external pipeline (FollowBench / InfoBench / AdvancedIF — same
three benchmarks the RL run uses).

## Convenience: chain fill → SFT

`chain_fill_then_sft.sh <FILL_PID>` waits for the fill PID to exit, sanity-
checks the parquet, and immediately launches SFT. Useful when you want to
leave the box and come back to a finished SFT. Pass the GPU/NPROC env
vars exactly like step 3.

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
    bash baselines/sft_distill/chain_fill_then_sft.sh <pid_of_fill_run>
```

## Files

| File | Purpose |
|---|---|
| `generate_teacher_responses.py` | Step 1 — offline vLLM teacher generation. |
| `fill_missing_responses.py` | Step 2 — retry truncated samples and merge. |
| `sft_distill_v1.0_base.yaml` | Hydra config for `verl.trainer.fsdp_sft_trainer`. |
| `run_sft_distill.sh` | Step 3 — torchrun launcher. |
| `chain_fill_then_sft.sh` | Optional fill-then-SFT chain. |
| `logs/` | Per-run training and generation logs. |
