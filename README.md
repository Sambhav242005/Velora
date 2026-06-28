# Velora

Velora is a from-scratch LLaMA-style decoder-only language model trainer and inference runtime for compact 80M-100M parameter models. It is pure single-process PyTorch: no DeepSpeed, no HuggingFace `Trainer`, and no hidden training framework.

The repository is designed for:

- local smoke tests before spending GPU time
- interruptible RunPod training with automatic resume
- long-context hybrid-attention experiments
- instruction and structured-output fine-tuning
- reproducible Hugging Face export bundles for custom PyTorch inference

Model checkpoints, tokenized datasets, and exported Hugging Face folders are intentionally not committed. They live under `out/`, `data/`, and `hf_repo/`, which are ignored by Git.

## Current status

The current polished path is the 100M structured-output checkpoint family, ending at `sft_100m_structured_strict_ctx16k`. The retained best checkpoint is exported with `scripts/export_hf_repo.py` into a self-contained runtime bundle containing:

- a slim inference checkpoint
- tokenizer
- `generator.py`
- minimal runtime source files
- `config.json`
- model-card README

Use the root repository for training, experimentation, and reproducibility. Use the exported `hf_repo/<repo-slug>/` folder for publishing model artifacts.

## Quick inference

Plain generation:

```bash
python generate.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --prompt "Cloud computing is" --max_new_tokens 80
```

Instruction generation:

```bash
python generate_instruct.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --instruction "What is artificial intelligence?" --max_new_tokens 120
```

Structured JSON generation uses parser-state guidance by default:

```bash
python generate_structured.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --format json --instruction "Return a JSON object with keys answer and confidence." --json_key_types answer:boolean,confidence:number
```

## Training log viewer

Open `docs/training-logbook.html` in a browser to inspect the included 10B-token 100M training run snapshot. The page is a static local-only dashboard for the provided run: it chunks dense training metrics for smoother charting, keeps validation loss exact, supports chart zoom ranges, compares train and validation loss on hover, highlights the best validation point, lists validation checks from start to final with nearby training parameters, and exports CSV/PNG.

## Quantization

Do not quantize the only published copy. Publish the normal slim checkpoint first so the model remains reproducible and easy to debug.

Quantization is useful as an additional inference artifact if you want a smaller CPU/demo download. For this 100M-class custom PyTorch model, the practical order is:

1. publish the slim full-precision export first
2. optionally add a separate inference-only quantized artifact later
3. label the quantized artifact clearly, because it is not meant for resume training

At this size, quantization is optional. It saves disk and RAM, but it is not required for GitHub readiness and can make debugging structured decoding harder.

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

Structured JSON generation uses a JSON parser-state guide automatically; no regex is required:

```bash
python generate_structured.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --format json --instruction "Return a JSON object with keys answer and confidence." --json_key_types answer:boolean,confidence:number
```

Regex-guided generation is still available as a lower-level escape hatch for simple custom output languages:

```bash
python generate_instruct.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --instruction "Answer yes or no: is 2+2=4?" --regex "\s*([Yy]es|[Nn]o)" --temperature 0
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
.\.venv\Scripts\python.exe generate_instruct.py --checkpoint out\<run>\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --instruction "What is artificial intelligence?" --max_new_tokens 120
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

## Hugging Face export

Create a self-contained Hugging Face model repo folder with the slim checkpoint, tokenizer, generator.py, minimal src/ runtime, requirements.txt, config.json, and a model-card README.md.

PowerShell:

    .\.venv\Scripts\python.exe scripts\export_hf_repo.py --repo_id sambhav24/velora-100m-structured-strict-ctx16k --checkpoint out\sft_100m_structured_strict_ctx16k\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --overwrite

This writes hf_repo/velora-100m-structured-strict-ctx16k/. That folder is ignored by Git because it contains model artifacts.

Before uploading, remove the stale files currently on the Hugging Face repo:

    .\.venv\Scripts\hf.exe repos delete-files sambhav24/velora-100m-structured-strict-ctx16k best.pt generate_structured.py velora_structured_strict_ctx16k_artifacts.tar src/

Then upload the generated bundle:

    .\.venv\Scripts\hf.exe upload sambhav24/velora-100m-structured-strict-ctx16k hf_repo\velora-100m-structured-strict-ctx16k .
