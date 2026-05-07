#!/usr/bin/env bash
# Wait for the running teacher-fill PID to exit, sanity-check the resulting
# parquet, then launch SFT distillation.
#
# Env vars:
#   CUDA_VISIBLE_DEVICES   GPUs to use for SFT (default: inherit from caller)
#   NPROC                  number of trainer processes (default: 4)
#
# Usage:
#   bash baselines/sft_distill/chain_fill_then_sft.sh <FILL_PID>
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
#       bash baselines/sft_distill/chain_fill_then_sft.sh <FILL_PID>

set -euo pipefail

FILL_PID="${1:?usage: $0 <FILL_PID>}"
NPROC="${NPROC:-4}"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
PARQUET="./datasets/sft_distill/wildchecklists_qwen3_8b_teacher.parquet"

echo "[chain] $(date +%H:%M:%S) waiting for fill PID ${FILL_PID} to finish..."
while kill -0 "${FILL_PID}" 2>/dev/null; do sleep 30; done
echo "[chain] $(date +%H:%M:%S) fill exited"

# Sanity check: parquet exists and has rows
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('${PARQUET}')
print(f'[chain] parquet rows: {len(df)}, prompt unique: {df[\"prompt\"].nunique()}')
assert len(df) > 0, 'empty parquet'
"

TS=$(date +%Y%m%d_%H%M%S)
SFT_LOG="${LOG_DIR}/sft_distill_${TS}.log"
echo "[chain] $(date +%H:%M:%S) launching SFT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherit>}, nproc=${NPROC}); log: ${SFT_LOG}"

NPROC="${NPROC}" nohup bash baselines/sft_distill/run_sft_distill.sh \
    trainer.n_gpus_per_node="${NPROC}" \
    > "${SFT_LOG}" 2>&1 &
SFT_PID=$!
echo "[chain] SFT PID=${SFT_PID}"
echo "[chain] log: ${SFT_LOG}"
