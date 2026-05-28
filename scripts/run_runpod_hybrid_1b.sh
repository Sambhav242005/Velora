#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/fineweb_processed_16k_1b"
TOKENIZER="tokenizer_fineweb_16k/tokenizer.json"
CONFIG="configs/runpod_sambhav_80m_v2_hybrid_1b.yaml"

if [[ ! -f "${DATA_DIR}/meta.json" ]]; then
  python scripts/prepare_fineweb.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config sample-10BT \
    --split train \
    --tokenizer "${TOKENIZER}" \
    --out_dir "${DATA_DIR}" \
    --max_tokens 1000000000 \
    --val_tokens 5000000 \
    --shard_tokens 10000000
fi

python train.py --config "${CONFIG}" --resume auto
