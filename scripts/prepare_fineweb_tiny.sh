#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_fineweb.py \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --split train \
  --tokenizer tokenizer/tokenizer.json \
  --out_dir data/fineweb_processed \
  --max_tokens 10000000 \
  --val_tokens 500000 \
  --shard_tokens 1000000
