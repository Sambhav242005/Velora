# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A from-scratch ~80M-parameter LLaMA-style decoder-only language model trainer ("Velora" / "sambhav-80m"). **Pure single-process PyTorch** -- no DeepSpeed, no HuggingFace `Trainer`. Designed to train safely on constrained / interruptible hardware (a local GPU, then RunPod). The code is intentionally simple and debuggable; keep it that way.

Pipeline: base pretrain -> continued pretraining (CPT) -> supervised fine-tuning (SFT). Everything is driven by YAML configs in `configs/`.

## Environment & setup

- **OS:** Windows (primary dev box). Shell examples use PowerShell; a bash tool is also available.
- **Python:** use the project venv -- do **not** assume a global `python`.
  - PowerShell: `.\.venv\Scripts\python.exe ...`
  - bash: `./.venv/Scripts/python.exe ...`
  - On RunPod/Linux: `source .venv/bin/activate` then `python ...`
- **PyTorch** (currently `2.11.0+cu128`) is installed **separately** from `requirements.txt` (via the CUDA index URL). `requirements.txt` covers everything else (numpy, pyyaml, tokenizers, datasets, tqdm, psutil, ...).
- **There is no automated test suite.** Do not run `pytest` expecting tests. Verify changes by the methods in "Verifying changes" below.

## Key commands

Run from the repo root.

```bash
# Inspect model size / token budget / resume state WITHOUT training:
./.venv/Scripts/python.exe train.py --config configs/<name>.yaml --info

# Base / CPT training (token-budget driven), with safe resume + tee'd logs:
./.venv/Scripts/python.exe train.py --config configs/<name>.yaml --resume auto --logs

# Instruction fine-tuning (step driven):
./.venv/Scripts/python.exe train_sft.py --config configs/<name>.yaml --resume auto --logs

# Prepare data (token shards for LM; arrays for SFT):
./.venv/Scripts/python.exe scripts/prepare_fineweb.py  --help   # FineWeb-edu LM shards
./.venv/Scripts/python.exe scripts/prepare_lm_hf.py    --help   # any HF dataset -> LM shards
./.venv/Scripts/python.exe scripts/prepare_sft.py      --help   # instruction SFT arrays
./.venv/Scripts/python.exe scripts/prepare_reasoning_mix.py --help

# Generate / chat:
./.venv/Scripts/python.exe generate.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --prompt "..."
./.venv/Scripts/python.exe generate_instruct.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --instruction "..."
./.venv/Scripts/python.exe generate_structured.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --format json --instruction "..."
./.venv/Scripts/python.exe generate_instruct.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --instruction "Answer yes or no: ..." --regex "\s*([Yy]es|[Nn]o)"
./.venv/Scripts/python.exe chat.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json   # interactive multi-turn

# Evaluate (GSM8K exact-match + loss-by-position):
./.venv/Scripts/python.exe scripts/eval_longctx.py --checkpoint out/<run>/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json
```

Useful `train.py` flags: `--max-steps`, `--max-tokens`, `--max-vram-gb`, `--max-vram-fraction`, `--max-micro-batch`, `--resume {auto|none|<path>}`, `--info`, `--logs`.

## Project structure

```
src/
  model.py       # GPT + ModelConfig: GQA, RoPE, RMSNorm, SwiGLU, tied embeddings, hybrid attention
  trainer.py     # Trainer: auto micro-batch finder, OOM recovery, checkpointing, resume, LR schedule
  data.py        # PackedMemmapDataset: memmapped token shards for pretraining/CPT
  sft_data.py    # SFTArrayDataset: fixed-length padded arrays with prompt-masked labels
  checkpoint.py  # atomic save, RNG state, resume-candidate discovery, milestone rotation
  memory.py      # RAM/VRAM monitoring + CUDA cleanup
  config.py      # YAML load + deep-merge helpers
  guided.py      # indexed regex automaton-guided logits masking
  json_guided.py # JSON parser-state guided logits masking
train.py / train_sft.py   # entry points (base/CPT and SFT)
generate.py / generate_instruct.py / generate_structured.py / chat.py   # inference
configs/         # one YAML per run (model + train + batch + checkpoint + eval blocks)
scripts/         # data prep + RunPod orchestration (*.ps1 / *.sh)
out/             # checkpoints (best.pt, final.pt, last.pt, milestone_*.pt, interrupted.pt, ...)
docs/            # PROJECT_NOTES.md (findings/bugs/decisions), runpod_longcontext_plan.md (runbook)
```

## Conventions

- **Config-driven.** Add a run = add a `configs/<name>.yaml`. Model block, optimizer, batch finder, checkpointing, and eval are all config keys. `vocab_size: auto` resolves from the tokenizer/dataset meta.
- **Checkpoints** are dicts (`model`, `optimizer`, `scaler`, `train_state`, `config`, `rng_state`, `data_rng_state`) written atomically. Loaded with `torch.load(..., weights_only=False)`. `out/<run>/last.pt` is always kept current; `best.pt` tracks lowest val loss.
- **Resume is automatic** (`--resume auto`): finds `last.pt`/`interrupted.pt`/`emergency.pt`/latest milestone/`best.pt`. Designed for spot-instance preemption (SIGTERM -> save -> resume).
- **Warm-start vs resume:** resume has priority and restores full training state. If no resume checkpoint is loaded, `train.base_checkpoint` loads *weights only* into a freshly-built model (new optimizer/schedule). SFT (`train_sft.py`) rebuilds the model from the checkpoint's stored config; `train.py` builds from the current YAML.
- Match the surrounding style: type hints, `from __future__ import annotations`, dataclasses, no external framework magic.
- **Secrets** (`HF_TOKEN`, `WANDB_API_KEY`) are read from environment variables only -- never hardcode them in code, configs, or git, and never `print` them (logs get uploaded). New code that needs a key must read it from `os.environ`. See `DEPLOYMENT.md`.

## Gotchas -- read before touching these

- **RoPE (`src/model.py`).** `rotate_half` uses the **half-split** convention and must stay paired with the `cat((freqs, freqs))` cos/sin cache. A previous version mixed half-split with an interleaved `rotate_half`, which broke relative-position encoding (see B1 in `docs/PROJECT_NOTES.md`). **Do not "simplify" `rotate_half` back to the `::2` / `1::2` interleaved form** -- it silently breaks RoPE. `rope_theta` is a `ModelConfig` field (default 10000); long-context runs raise it (e.g., 500000).
- **Checkpoint compatibility.** Changing positional encoding (RoPE form or `rope_theta`) makes older checkpoints incompatible. The current `v3_*` configs warm-start from the pre-fix 1B base on purpose (with a heal phase) -- see `docs/PROJECT_NOTES.md` section 4.
- **Long-context memory (B2).** `sliding`/`csa`/`hca` attention build explicit O(T^2) masks (`src/model.py` `_local_mask`, `_compressed_attention`); only `full` uses memory-efficient FlashAttention. Practical context ceiling is ~16K until those paths are migrated to FlexAttention. Don't assume long context is free.
- **Hybrid `torch.compile`.** The masked `sliding`/`csa`/`hca` helpers are intentionally wrapped with `torch.compiler.disable` / `torch._dynamo.disable` because Inductor cannot reliably trace their dynamic mask/block logic. Keep those graph-break wrappers unless the helpers are rewritten with FlexAttention. The attention kind is resolved **once in `__init__`** as `self._kind` -- do **not** re-derive it from `self.layer_idx` inside `forward`, or Dynamo specializes `forward` per layer index, blows `recompile_limit`, and falls the whole frame (projections included) back to eager (B12). `use_flex_attention` is opt-in for sliding layers only and should fall back to eager unless the fused Triton-compiled FlexAttention path is available.
- **Hybrid attention pattern.** The current default is final-global: `full,sliding,csa,hca,sliding,csa,hca,full`. Keep future long-context configs ending on a `full` layer unless you are deliberately running an ablation.
- **No document-boundary masking** in `PackedMemmapDataset` (B3) -- windows cross document boundaries.

## Verifying changes (no test suite)

1. **Config/model sanity:** load each touched config and build the model:
   ```bash
   ./.venv/Scripts/python.exe -c "from src.config import load_yaml; from src.model import ModelConfig, GPT; c=load_yaml('configs/<name>.yaml'); m=dict(c['model']); m['vocab_size']=16000; print(GPT(ModelConfig(**m)).num_parameters()/1e6,'M')"
   ```
2. **Budget/resume check:** `train.py --config <cfg> --info`.
3. **Smoke run:** `train.py --config <cfg> --max-steps 20 --resume none` and confirm finite, decreasing loss.
4. **RoPE correctness** (if you touch attention/RoPE): build a `CausalSelfAttention`, apply RoPE to a fixed `(q,k)` across positions, and confirm `<rope(q,m), rope(k,n)>` is constant along each `m-n` diagonal (spread ~ 0). See `docs/PROJECT_NOTES.md` B1 for the exact check.

## Updating the project (configs, docs, code)

**Configs are the source of truth; docs describe them. When they disagree, fix the doc.** Whenever you change behavior, update the artifacts below in the *same* change so they never drift.

**YAML configs (`configs/*.yaml`)**
- Blocks: `project_name`, `seed`, `out_dir`, `data`, `model`, `train`, `batch`, `checkpoint`, `eval` (plus `sft` for SFT runs). Loaded by `src/config.py`, then `ModelConfig(**config["model"])`.
- **Add a new run:** copy the closest existing config and change `project_name`, `out_dir` (must be unique -- resume keys off it), `data.data_dir`, and the knobs you care about. Chain stages with `train.base_checkpoint` (LM/CPT) or `sft.base_checkpoint` (SFT).
- Keep `vocab_size: auto` (resolves from the dataset/tokenizer meta).
- Every key under `model:` must be a field on `ModelConfig` -- unknown keys raise `TypeError`; missing keys use dataclass defaults.
- After editing a config, validate it (see "Verifying changes"): build the model from it and run `--info`.

**Model code (`src/model.py` -> `ModelConfig`)**
- Adding a hyperparameter: add a dataclass field **with a default** so existing configs and saved checkpoints still load; then wire it where used.
- Renaming/removing a field: update **all** `configs/*.yaml` in the same change.
- Anything that changes positional encoding (RoPE form or `rope_theta`) **invalidates existing checkpoints** -- flag it (see Gotchas) and record it in `docs/PROJECT_NOTES.md`.

**Docs (`*.md`)**
- `AGENTS.md` (this file): update when commands, structure, conventions, or gotchas change.
- `docs/PROJECT_NOTES.md`: when you find or fix a bug, add a `B#` entry (*what / where `file:line` / evidence / impact / status*) and update its summary table; record notable choices in section 4 Decisions; update the section 8 changelog.
- `docs/runpod_longcontext_plan.md`: update whenever configs, datasets, or commands change so it stays runnable end-to-end.

**Scripts & data prep (`scripts/`)**
- Follow the existing argparse pattern: explicit flags, `--overwrite`, a written `meta.json`, shard writers for token data. For a new dataset prefer `prepare_lm_hf.py` (LM) or `prepare_sft.py` (instruction) -- they auto-detect common field layouts.

**Do not commit generated artifacts.** `out/` (checkpoints), `data/` (tokenized shards), `hf_repo/` (export bundles), `hf_models/` (downloaded bundles), `.venv/`, and `*.pt`/`*.npy`/`*.bin` are already covered by `.gitignore` -- keep them ignored; don't force-add them.


## Portfolio Cover Asset

Maintain a project-specific SVG at `docs/portfolio-cover.svg`.

Rules:
- The SVG must be hand-authored/static, not a raster screenshot, AI-generated image, base64 image, or external asset.
- Use `width="1200"`, `height="760"`, `viewBox="0 0 1200 760"`.
- It should visually summarize the real current project: architecture, workflow, UI, model pipeline, or system behavior.
- Update this SVG whenever major project functionality, architecture, or branding changes.
- Keep text minimal and readable at thumbnail size.
- No fake product names, unrelated placeholder visuals, or generic charts.
- The portfolio repo may copy this file into `public/project-assets` as the local backup/rendering copy.

## Git / workflow

- Default branch is `main`. Prefer a feature branch for changes.
- **Commit or push only when the user asks.** End commit messages with the required co-author trailer if committing.

## More context

- `DEPLOYMENT.md` -- secrets, artifact storage (HF Hub), inference, and serving.
- `TRAINING.md` -- concise setup, training, SFT, generation, RunPod, and HF publish commands.
- `docs/PROJECT_NOTES.md` -- full findings, the bug register (B1-B8), and design decisions.
- `docs/runpod_longcontext_plan.md` -- the end-to-end RunPod training runbook.

## Hugging Face export workflow

Use scripts/export_hf_repo.py when publishing this custom PyTorch model to Hugging Face. It creates hf_repo/<repo-slug>/ with:

- checkpoints/model.pt slim inference checkpoint
- tokenizer/tokenizer.json
- generator.py
- minimal src/ runtime files
- requirements.txt
- config.json
- model-card README.md

Example:

    ./.venv/Scripts/python.exe scripts/export_hf_repo.py --repo_id sambhav24/velora-100m-structured-strict-ctx16k --checkpoint out/sft_100m_structured_strict_ctx16k/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --overwrite
    ./.venv/Scripts/hf.exe repos delete-files sambhav24/velora-100m-structured-strict-ctx16k best.pt generate_structured.py velora_structured_strict_ctx16k_artifacts.tar src/
    ./.venv/Scripts/hf.exe upload sambhav24/velora-100m-structured-strict-ctx16k hf_repo/velora-100m-structured-strict-ctx16k .

Do not upload only best.pt for public/useful model repos; include the runtime bundle so inference is reproducible. Delete stale remote files before uploading so old root-level artifacts do not remain visible.
