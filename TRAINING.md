# Training

This file is for training and local experimentation. The short user-facing model download and usage steps are in [README.md](README.md).

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
```

Install PyTorch for your CUDA version from the official PyTorch command, then install the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Smoke Test

Run the smallest config first:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\debug_12m.yaml
```

Stop with Ctrl+C to confirm it saves `interrupted.pt` and `last.pt`.

Resume:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\debug_12m.yaml --resume auto
```

## Prepare Local Sample Data

Create tiny sample text:

```powershell
.\.venv\Scripts\python.exe scripts\make_sample_data.py
```

Train a small tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts\train_tokenizer.py --input_dir data\raw --out_dir tokenizer --vocab_size 4096
```

Tokenize the sample data:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --input_dir data\raw --tokenizer tokenizer\tokenizer.json --out_dir data\processed --val_fraction 0.05
```

## Train A Local Model

Inspect a config without training:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\local_fineweb_80m_tiny_16k.yaml --info
```

Train:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\local_fineweb_80m_tiny_16k.yaml --resume auto --logs
```

Train for a fixed number of steps:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\local_fineweb_80m_tiny_16k.yaml --max-steps 10000 --resume auto --logs
```

Use a VRAM cap:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\local_fineweb_80m_tiny_16k.yaml --max-vram-gb 10 --max-micro-batch 64 --resume auto --logs
```

## FineWeb-Edu Data

Export text for tokenizer training:

```powershell
.\.venv\Scripts\python.exe scripts\export_fineweb_text.py --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --split train --out_file data\tokenizer_corpus\fineweb_sample.txt --max_chars 200000000 --overwrite
```

Train a 16k tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts\train_tokenizer.py --input_dir data\tokenizer_corpus --out_dir tokenizer_fineweb_16k --vocab_size 16000
```

Prepare token shards:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_fineweb.py --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --split train --tokenizer tokenizer_fineweb_16k\tokenizer.json --out_dir data\fineweb_processed_16k --max_tokens 10000000 --val_tokens 500000 --shard_tokens 1000000 --overwrite
```

## Instruction Fine-Tuning

Prepare Alpaca-cleaned:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sft.py --dataset_name yahma/alpaca-cleaned --tokenizer tokenizer_fineweb_16k\tokenizer.json --out_dir data\sft\alpaca_cleaned_512 --max_seq_len 512 --overwrite
```

Train Alpaca SFT:

```powershell
.\.venv\Scripts\python.exe train_sft.py --config configs\sft_alpaca_80m.yaml --resume auto --logs
```

Prepare Dolly:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sft.py --dataset_name databricks/databricks-dolly-15k --tokenizer tokenizer_fineweb_16k\tokenizer.json --out_dir data\sft\dolly_512 --max_seq_len 512 --overwrite
```

Train Dolly SFT:

```powershell
.\.venv\Scripts\python.exe train_sft.py --config configs\sft_dolly_80m.yaml --resume auto --logs
```

## Generate From A Local Checkpoint

Plain generation:

```powershell
.\.venv\Scripts\python.exe generate.py --checkpoint out\<run>\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --prompt "Cloud computing is" --max_new_tokens 80
```

Instruction generation:

```powershell
.\.venv\Scripts\python.exe generate_instruct.py --checkpoint out\<run>\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --instruction "What is artificial intelligence?" --max_new_tokens 120
```

Structured JSON generation:

```powershell
.\.venv\Scripts\python.exe generate_structured.py --checkpoint out\<run>\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --format json --instruction "Return a JSON object with keys answer and confidence." --json_key_types answer:boolean,confidence:number
```

## RunPod

For the included v2 hybrid 80M 1B-token RunPod path:

```bash
bash scripts/run_runpod_hybrid_1b.sh
```

For a direct config run:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\runpod_80m_1b.yaml --resume auto --logs
```

Keep checkpoints on persistent storage when using RunPod.

## Publish To Hugging Face

Export the current structured SFT model:

```powershell
.\.venv\Scripts\python.exe scripts\export_hf_repo.py --repo_id sambhav24/velora-100m-structured-strict-ctx16k --checkpoint out\sft_100m_structured_strict_ctx16k\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --overwrite
```

Upload the generated bundle:

```powershell
.\.venv\Scripts\hf.exe upload sambhav24/velora-100m-structured-strict-ctx16k hf_repo\velora-100m-structured-strict-ctx16k .
```

Do not upload only `best.pt`. Upload the full generated bundle so the model can run from Hugging Face with its custom runtime.
