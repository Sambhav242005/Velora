# Velora

This project is made for **local smoke testing first** and then a larger RunPod training run later.

It includes:

- LLaMA-style decoder-only transformer
- RoPE, RMSNorm, SwiGLU, GQA
- PyTorch SDPA / FlashAttention path when available
- RAM-safe memmap dataset loading
- VRAM logging
- automatic micro-batch finder
- optional auto batch increase/decrease
- checkpoint save on interval, crash, OOM, Ctrl+C, and SIGTERM
- atomic checkpoint writing to avoid corrupt saves
- automatic resume from `last.pt`
- tiny local token budget profiles

## 0. Install

Create venv:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
pip install -U pip
```

Install PyTorch for your CUDA version from the official PyTorch command, then:

```bash
pip install -r requirements.txt
```

Example CUDA install command may look like:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 1. Make tiny sample data

```bash
python scripts/make_sample_data.py
```

This creates `data/raw/sample.txt` only for testing the pipeline.

## 2. Train tokenizer

For local testing use smaller vocab to make the model lighter:

```bash
python scripts/train_tokenizer.py --input_dir data/raw --out_dir tokenizer --vocab_size 4096
```

For serious run later, use 16000 or 32000 vocab.

## 3. Tokenize dataset into memmap binary

```bash
python scripts/prepare_dataset.py --input_dir data/raw --tokenizer tokenizer/tokenizer.json --out_dir data/processed --val_fraction 0.05
```

This writes:

```text
data/processed/train.bin
data/processed/val.bin
data/processed/meta.json
```

## Training with FineWeb-Edu

Install:

```bash
pip install -r requirements.txt
```

For a real FineWeb tokenizer, first stream a bounded raw-text corpus:

```bash
python scripts/export_fineweb_text.py --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --split train --out_file data/tokenizer_corpus/fineweb_sample.txt --max_chars 200000000 --overwrite
```

Then train a tokenizer from that corpus:

```bash
python scripts/train_tokenizer.py --input_dir data/tokenizer_corpus --out_dir tokenizer_fineweb_16k --vocab_size 16000
```

Prepare FineWeb-Edu tokens using the new tokenizer:

```bash
python scripts/prepare_fineweb.py --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --split train --tokenizer tokenizer_fineweb_16k/tokenizer.json --out_dir data/fineweb_processed_16k --max_tokens 10000000 --val_tokens 500000 --shard_tokens 1000000 --overwrite
```

Train local tiny:

```bash
python train.py --config configs/local_fineweb_80m_tiny_16k.yaml
```

Train the v2 hybrid 80M model for a 1B-token RunPod run:

```bash
bash scripts/run_runpod_hybrid_1b.sh
```

This uses `configs/runpod_sambhav_80m_v2_hybrid_1b.yaml`, prepares
`data/fineweb_processed_16k_1b` if it is missing, and resumes from
`out/runpod_sambhav_80m_v2_hybrid_1b/last.pt` when available.

Train for a fixed number of optimizer steps, for example 10k steps:

```bash
python train.py --config configs/local_fineweb_80m_tiny_16k.yaml --max-steps 10000 --resume auto
```

You can also put this in the config under `train.max_steps`. When `max_steps` is set, it is the stopping target instead of `max_tokens`.

To let the trainer use a specific VRAM budget, set an absolute cap and a high enough micro-batch search ceiling:

```bash
python train.py --config configs/local_fineweb_80m_tiny_16k.yaml --max-steps 10000 --resume auto --max-vram-gb 10 --max-micro-batch 64
```

The auto batch finder picks the largest micro-batch whose peak reserved VRAM stays under the cap, then adjusts gradient accumulation from `target_tokens_per_update`.

Inspect model size and token budget without training:

```bash
python train.py --config configs/local_fineweb_80m_tiny_16k.yaml --info
```

That config points to:

```yaml
data:
  tokenizer_path: tokenizer_fineweb_16k/tokenizer.json
  data_dir: data/fineweb_processed_16k
```

Resume:

```bash
python train.py --config configs/local_fineweb_80m_tiny_16k.yaml --resume auto
```

Generate:

```bash
python generate.py --checkpoint out/local_fineweb_80m_tiny_16k_80m/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --prompt "Cloud computing is" --max_new_tokens 80
```

## Instruction fine-tuning

The base model is a text completer. To make a separate instruction-tuned model, first prepare SFT arrays. The recommended first dataset is `yahma/alpaca-cleaned` because it is a cleaned 52k instruction/input/output dataset with a simple schema. A second comparison config is included for `databricks/databricks-dolly-15k`.

Prepare Alpaca-cleaned:

```bash
.\.venv\Scripts\python.exe scripts\prepare_sft.py --dataset_name yahma/alpaca-cleaned --tokenizer tokenizer_fineweb_16k\tokenizer.json --out_dir data\sft\alpaca_cleaned_512 --max_seq_len 512 --overwrite
```

Train a separate Alpaca SFT checkpoint folder:

```bash
.\.venv\Scripts\python.exe train_sft.py --config configs\sft_alpaca_80m.yaml --resume auto
```

Prepare Dolly:

```bash
.\.venv\Scripts\python.exe scripts\prepare_sft.py --dataset_name databricks/databricks-dolly-15k --tokenizer tokenizer_fineweb_16k\tokenizer.json --out_dir data\sft\dolly_512 --max_seq_len 512 --overwrite
```

Train a separate Dolly SFT checkpoint folder:

```bash
.\.venv\Scripts\python.exe train_sft.py --config configs\sft_dolly_80m.yaml --resume auto
```

Test an SFT model:

```bash
.\.venv\Scripts\python.exe generate_instruct.py --checkpoint out\sft_alpaca_80m\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --instruction "What is artificial intelligence?" --max_new_tokens 120
```

## 4. Local smoke test

First test the full safety system with a small model:

```bash
python train.py --config configs/debug_12m.yaml
```

Then test the 80M model with very low tokens:

```bash
python train.py --config configs/local_80m_tiny.yaml
```

You can stop with Ctrl+C. It should save `interrupted.pt` and `last.pt`.

Resume automatically:

```bash
python train.py --config configs/local_80m_tiny.yaml --resume auto
```

## 5. Generate sample text

```bash
python generate.py --checkpoint out/local_80m_tiny/last.pt --tokenizer tokenizer/tokenizer.json --prompt "Cloud computing is" --max_new_tokens 80
```

## 6. Later RunPod command

After local tests pass:

```bash
python train.py --config configs/runpod_80m_1b.yaml --resume auto
```

## Recommended order

1. Run `debug_12m.yaml`
2. Test Ctrl+C save
3. Resume from checkpoint
4. Run `local_80m_tiny.yaml`
5. Only then use RunPod

## Notes

- This code is intentionally simple and debuggable.
- The local sample dataset is not enough to make a useful model.
- For a real model, replace `data/raw/sample.txt` with clean dataset text files.
- Keep checkpoints on a persistent disk/network volume when using RunPod.
