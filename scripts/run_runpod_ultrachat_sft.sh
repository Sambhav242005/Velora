#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/sft/ultrachat_768"
TOKENIZER="tokenizer_fineweb_16k/tokenizer.json"
CONFIG="configs/sft_ultrachat_runpod_v2.yaml"

if [[ ! -f "${DATA_DIR}/meta.json" ]]; then
  python scripts/prepare_sft.py \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --split train_sft \
    --tokenizer "${TOKENIZER}" \
    --out_dir "${DATA_DIR}" \
    --max_seq_len 768 \
    --val_fraction 0.02 \
    --streaming true \
    --overwrite
fi

python train_sft.py --config "${CONFIG}" --resume auto
