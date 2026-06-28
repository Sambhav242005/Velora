# Velora

Velora is a small custom PyTorch language model project. It includes training code, inference scripts, and a static training-log viewer.

This is not a Hugging Face `transformers` model. It uses the custom files in this repo, especially `generator.py`, `src/`, `checkpoints/model.pt`, and `tokenizer/tokenizer.json`.

## Download The Model

Model page: [sambhav24/velora-100m-structured-strict-ctx16k](https://huggingface.co/sambhav24/velora-100m-structured-strict-ctx16k)

Download the whole repo bundle, not only `model.pt`.

Website download option: open [Files and versions](https://huggingface.co/sambhav24/velora-100m-structured-strict-ctx16k/tree/main), download the files, then paste/extract them into the folder shown below.

Download with the Hugging Face CLI:

```powershell
.\.venv\Scripts\hf.exe download sambhav24/velora-100m-structured-strict-ctx16k --local-dir hf_models\velora-100m-structured-strict-ctx16k
```

If you download from the website instead, paste/extract the downloaded model folder here:

```text
hf_models/velora-100m-structured-strict-ctx16k/
```

That folder should contain:

```text
generator.py
checkpoints/model.pt
tokenizer/tokenizer.json
src/
requirements.txt
config.json
```

`hf_models/` is ignored by Git, so the downloaded model will not be committed.

## Use The Model

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r hf_models\velora-100m-structured-strict-ctx16k\requirements.txt
```

Run JSON-guided generation:

```powershell
.\.venv\Scripts\python.exe hf_models\velora-100m-structured-strict-ctx16k\generator.py --checkpoint hf_models\velora-100m-structured-strict-ctx16k\checkpoints\model.pt --tokenizer hf_models\velora-100m-structured-strict-ctx16k\tokenizer\tokenizer.json --format json --instruction "Return a JSON object with keys answer and confidence. Answer whether 2+2=4." --json_keys answer,confidence --json_key_types answer:boolean,confidence:number --temperature 0
```

Run yes/no constrained generation:

```powershell
.\.venv\Scripts\python.exe hf_models\velora-100m-structured-strict-ctx16k\generator.py --checkpoint hf_models\velora-100m-structured-strict-ctx16k\checkpoints\model.pt --tokenizer hf_models\velora-100m-structured-strict-ctx16k\tokenizer\tokenizer.json --format json --instruction "Answer yes or no: is 2+2=4?" --regex "\s*([Yy]es|[Nn]o)" --temperature 0
```

## Train

Training instructions are in [TRAINING.md](TRAINING.md).

Quick local smoke test:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\debug_12m.yaml
```

Resume a run:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\<config>.yaml --resume auto --logs
```

## Training Log Viewer

Open this file in a browser:

```text
docs/training-logbook.html
```

It shows the provided training log with train/validation views, zoom, best validation highlight, and CSV/PNG export.

## Hugging Face Export

To publish the current best model bundle:

```powershell
.\.venv\Scripts\python.exe scripts\export_hf_repo.py --repo_id sambhav24/velora-100m-structured-strict-ctx16k --checkpoint out\sft_100m_structured_strict_ctx16k\best.pt --tokenizer tokenizer_fineweb_16k\tokenizer.json --overwrite
.\.venv\Scripts\hf.exe upload sambhav24/velora-100m-structured-strict-ctx16k hf_repo\velora-100m-structured-strict-ctx16k .
```

`hf_repo/` is ignored by Git because it contains generated model artifacts.

## What Not To Commit

These are ignored and should stay out of Git:

- `out/`
- `data/`
- `hf_repo/`
- `hf_models/`
- `.venv/`
- `*.pt`, `*.bin`, `*.npy`

## Quantization

Do not quantize the only published copy. Publish the normal slim checkpoint first. Quantization can be added later as a separate inference-only artifact.
