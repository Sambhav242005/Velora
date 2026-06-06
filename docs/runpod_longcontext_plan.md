# RunPod Long-Context 80M Training Plan (v3)

> **For agentic workers:** steps use checkbox (`- [ ]`) syntax. Phases run in order; each produces a usable checkpoint on its own. Local code/config work happens in Phase 0, then everything else runs on RunPod.

**Goal:** Train a coherent ~80M LLaMA-style model with a **16K-token context window**, end-to-end on RunPod, with a clean curriculum: base pretrain → domain CPT → context extension → instruction tuning.

**Architecture:** Two-phase context strategy — pretrain the bulk at 2K (cheap), then a short continued-pretraining phase at 16K on long documents to teach long-range positions/dependencies. RoPE is now correct (fixed) and uses a high base frequency (`rope_theta=500000`) tuned for long context. The hybrid attention pattern keeps local/compressed layers for coverage but starts and ends each 24-layer stack with `full` global attention.

**Tech stack:** PyTorch 2.11 (bf16, SDPA/FlashAttention, gradient checkpointing), HuggingFace `datasets` (streaming), the repo's `train.py` / `train_sft.py`, single rented GPU (24–48 GB).

---

## Decisions locked in this plan (change here if you disagree)

| Decision | Value | Why |
|---|---|---|
| Target context | **16K** | Big jump from 2K, fits a 24–48 GB GPU without an attention rewrite. 32K+ needs the FlexAttention task (Phase 8, optional). |
| Base init | **Warm-start from your 1B** | Load `runpod_sambhav_80m_v2_hybrid_1b/final.pt` as initialization (not resume); attention re-heals over the first chunk of training. |
| Hybrid pattern | **final-global** | Use `full,sliding,csa,hca,sliding,csa,hca,full` so each 24-layer stack ends on a precise global layer while keeping 6 full layers total. |
| `rope_theta` | **10000 for 2K stages → 500000 at 16K** | Heal the rotation fix first at the frequency the 1B was trained on; raise theta only at the context-extension stage. |
| Base pretrain tokens | **+4B (≈5B effective)** | Warm-started from 1B; 4B more clean tokens fixes the under-training. |
| Context-extension tokens | **~500M** on long books (PG19) | Extension needs *long* docs, not packed short ones. |
| GPU | 24 GB for 2K stages; **48 GB for the 16K stage** | 16K masks are memory-heavy (see Risks). |
| Checkpoint storage | RunPod **persistent volume** at `/workspace` | Survives pod stop/restart and spot preemption. |

All new artifacts are prefixed `v3` so nothing collides with your existing `out/` and `configs/`.

---

## Phase 0 — Local code & config prep (do before renting any GPU)

### Task 0.1 — Confirm the RoPE fix is in place ✅ (already done)

**Files:** `src/model.py`

- [x] `rotate_half` uses the half-split convention (`model.py:43`)
- [x] `ModelConfig` has `rope_theta: float = 10000.0` (`model.py:30`)
- [x] frequency table uses `config.rope_theta` (`model.py:73`)

- [ ] **Verify it imports and the relative-position property holds.** Run from repo root:

```bash
.venv/Scripts/python.exe -c "import torch; from src.model import ModelConfig, CausalSelfAttention, apply_rope; c=ModelConfig(vocab_size=16,n_embd=256,n_head=8,n_kv_head=8,block_size=128,rope_theta=500000.0); a=CausalSelfAttention(c,0); cos,sin=a._rope_cache(64,torch.device('cpu'),torch.float32); g=torch.Generator().manual_seed(0); q=torch.randn(a.head_dim,generator=g); k=torch.randn(a.head_dim,generator=g); Q=q.view(1,1,1,-1).expand(1,64,1,-1).contiguous(); K=k.view(1,1,1,-1).expand(1,64,1,-1).contiguous(); S=(apply_rope(Q,cos,sin)[0,:,0,:]@apply_rope(K,cos,sin)[0,:,0,:].T); import itertools; print('max diag spread', max((S.diagonal(o).max()-S.diagonal(o).min()).item() for o in range(-63,64) if S.diagonal(o).numel()>1))"
```

Expected: `max diag spread` ≈ `1e-5` or smaller (machine noise → RoPE correct).

### Task 0.2 — Create the base pretrain config

**Files:** Create `configs/v3_base_2k.yaml`

- [ ] Write the file:

```yaml
project_name: v3-base-2k
seed: 1337
out_dir: out/v3_base_2k

data:
  data_dir: data/fineweb_16k_5b
  tokenizer_path: tokenizer_fineweb_16k/tokenizer.json
  num_workers: 0

model:
  vocab_size: auto
  n_layer: 24
  n_embd: 512
  n_head: 8
  n_kv_head: 2
  block_size: 2048
  dropout: 0.0
  bias: false
  use_gradient_checkpointing: true
  attention_mode: hybrid
  hybrid_attention_pattern: full,sliding,csa,hca,sliding,csa,hca,full
  sliding_window: 512
  csa_block_size: 64
  csa_local_window: 512
  hca_block_size: 256
  hca_local_window: 512
  rope_theta: 500000.0

train:
  max_tokens: 5_000_000_000
  dtype: bf16
  compile: false
  learning_rate: 0.0003
  min_lr: 0.00003
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  warmup_steps: 2000
  grad_clip: 1.0
  target_tokens_per_update: 131072
  log_interval: 20
  eval_every_tokens: 25_000_000
  save_every_tokens: 100_000_000
  save_every_minutes: 20

batch:
  auto_find: true
  start_micro_batch: 1
  max_micro_batch: 64
  reduce_on_oom: true
  increase_if_safe: true
  increase_every_steps: 200
  max_vram_fraction: 0.90

checkpoint:
  resume: auto
  keep_last_n: 3
  save_atomic: true
  save_on_interrupt: true
  save_on_exception: true

eval:
  iters: 50
  batch_size: 4
```

### Task 0.3 — Create the math CPT config

**Files:** Create `configs/v3_cpt_math_2k.yaml`

- [ ] Write the file (same `model` block as Task 0.2):

```yaml
project_name: v3-cpt-math-2k
seed: 1337
out_dir: out/v3_cpt_math_2k

data:
  data_dir: data/lm/openwebmath_2k
  tokenizer_path: tokenizer_fineweb_16k/tokenizer.json
  num_workers: 0

model:
  vocab_size: auto
  n_layer: 24
  n_embd: 512
  n_head: 8
  n_kv_head: 2
  block_size: 2048
  dropout: 0.0
  bias: false
  use_gradient_checkpointing: true
  attention_mode: hybrid
  hybrid_attention_pattern: full,sliding,csa,hca,sliding,csa,hca,full
  sliding_window: 512
  csa_block_size: 64
  csa_local_window: 512
  hca_block_size: 256
  hca_local_window: 512
  rope_theta: 500000.0

train:
  base_checkpoint: out/v3_base_2k/final.pt
  max_tokens: 1_000_000_000
  dtype: bf16
  compile: false
  learning_rate: 0.0001
  min_lr: 0.00001
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  warmup_steps: 200
  grad_clip: 1.0
  target_tokens_per_update: 131072
  log_interval: 20
  eval_every_tokens: 25_000_000
  save_every_tokens: 100_000_000
  save_every_minutes: 30

batch:
  auto_find: true
  auto_find_on_resume: true
  start_micro_batch: 1
  max_micro_batch: 64
  reduce_on_oom: true
  increase_if_safe: true
  increase_every_steps: 200
  max_vram_fraction: 0.90

checkpoint:
  resume: auto
  keep_last_n: 3
  save_atomic: true
  save_on_interrupt: true
  save_on_exception: true

eval:
  iters: 30
  batch_size: 4
```

### Task 0.4 — Create the web CPT config

**Files:** Create `configs/v3_cpt_web_2k.yaml`

- [ ] Same as Task 0.3 with these differences:
  - `project_name: v3-cpt-web-2k`
  - `out_dir: out/v3_cpt_web_2k`
  - `data.data_dir: data/lm/ultrafineweb_2k`
  - `train.base_checkpoint: out/v3_cpt_math_2k/final.pt`
  - everything else (model block, optimizer, batch, checkpoint, eval) identical to Task 0.3.

### Task 0.5 — Create the context-extension (16K) config

**Files:** Create `configs/v3_ctx16k.yaml`

- [ ] Write the file (note the larger `block_size` and scaled attention windows):

```yaml
project_name: v3-ctx16k
seed: 1337
out_dir: out/v3_ctx16k

data:
  data_dir: data/lm/pg19_16k
  tokenizer_path: tokenizer_fineweb_16k/tokenizer.json
  num_workers: 0

model:
  vocab_size: auto
  n_layer: 24
  n_embd: 512
  n_head: 8
  n_kv_head: 2
  block_size: 16384
  dropout: 0.0
  bias: false
  use_gradient_checkpointing: true
  attention_mode: hybrid
  hybrid_attention_pattern: full,sliding,csa,hca,sliding,csa,hca,full
  sliding_window: 2048
  csa_block_size: 256
  csa_local_window: 2048
  hca_block_size: 1024
  hca_local_window: 2048
  rope_theta: 500000.0

train:
  base_checkpoint: out/v3_cpt_web_2k/final.pt
  max_tokens: 500_000_000
  dtype: bf16
  compile: false
  learning_rate: 0.00005
  min_lr: 0.000005
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  warmup_steps: 200
  grad_clip: 1.0
  target_tokens_per_update: 131072
  log_interval: 10
  eval_every_tokens: 20_000_000
  save_every_tokens: 50_000_000
  save_every_minutes: 20

batch:
  auto_find: true
  auto_find_on_resume: true
  start_micro_batch: 1
  max_micro_batch: 8
  reduce_on_oom: true
  increase_if_safe: false
  max_vram_fraction: 0.90

checkpoint:
  resume: auto
  keep_last_n: 2
  save_atomic: true
  save_on_interrupt: true
  save_on_exception: true

eval:
  iters: 10
  batch_size: 1
```

### Task 0.6 — Create the SFT configs

**Files:** Create `configs/v3_sft_chat.yaml` and `configs/v3_sft_reasoning.yaml`

- [ ] `configs/v3_sft_chat.yaml` — base = the 16K checkpoint, moderate SFT length (4K) to preserve long context:

```yaml
project_name: v3-sft-chat
seed: 1337
out_dir: out/v3_sft_chat

data:
  data_dir: data/sft/ultrachat_4k
  tokenizer_path: tokenizer_fineweb_16k/tokenizer.json

model:
  vocab_size: auto
  n_layer: 24
  n_embd: 512
  n_head: 8
  n_kv_head: 2
  block_size: 16384
  dropout: 0.0
  bias: false
  use_gradient_checkpointing: true
  attention_mode: hybrid
  hybrid_attention_pattern: full,sliding,csa,hca,sliding,csa,hca,full
  sliding_window: 2048
  csa_block_size: 256
  csa_local_window: 2048
  hca_block_size: 1024
  hca_local_window: 2048
  rope_theta: 500000.0

sft:
  base_checkpoint: out/v3_ctx16k/final.pt
  max_steps: 1800

train:
  dtype: bf16
  learning_rate: 0.00002
  min_lr: 0.000002
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.95
  warmup_steps: 80
  grad_clip: 1.0
  target_tokens_per_update: 32768
  log_interval: 10
  eval_interval_steps: 50
  save_interval_steps: 150

batch:
  auto_find: true
  auto_find_on_resume: true
  start_micro_batch: 1
  max_vram_fraction: 0.90
  max_micro_batch: 32

checkpoint:
  resume: auto
  save_atomic: true
  save_on_interrupt: true
  save_on_exception: true

eval:
  iters: 50
  batch_size: 4
```

- [ ] `configs/v3_sft_reasoning.yaml` — identical to `v3_sft_chat.yaml` except:
  - `project_name: v3-sft-reasoning`
  - `out_dir: out/v3_sft_reasoning`
  - `data.data_dir: data/sft/reasoning_polish_1024`
  - `sft.base_checkpoint: out/v3_sft_chat/best.pt`
  - `sft.max_steps: 500`
  - `train.learning_rate: 0.000005`, `train.min_lr: 0.0000005`, `train.warmup_steps: 30`

> Note: the model `block_size` stays 16384 in SFT so the long-context positions keep getting exercised; the SFT *data* `max_seq_len` (4096 / 1024) just controls example length.

### Task 0.7 — Add a long-context eval script

**Files:** Create `scripts/eval_longctx.py`

- [ ] Write a script that measures (a) GSM8K exact-match and (b) per-position validation loss (does loss keep dropping deep into the 16K window?). Full content:

```python
from __future__ import annotations
import argparse, re, sys
import numpy as np
import torch
from tokenizers import Tokenizer
sys.path.insert(0, ".")
from src.model import GPT, ModelConfig
from generate_instruct import format_prompt, extract_response

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"]["model"])
    cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
    model = GPT(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model

def gsm8k_em(model, tok, device, n=200):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    correct = 0
    for ex in ds:
        prompt = format_prompt(ex["question"], "")
        ids = tok.encode(prompt, add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=256, temperature=0.0, top_k=0, top_p=None)
        out = extract_response(tok.decode(y[0].tolist()))
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        pred = re.findall(r"-?\d[\d,]*", out)
        pred = pred[-1].replace(",", "") if pred else ""
        correct += int(pred == gold)
    return correct / max(1, len(ds))

def ppl_by_position(model, tok, device, val_npy, block, bins=8):
    data = np.load(val_npy, mmap_mode="r")
    n = (len(data) // (block + 1)) * (block + 1)
    chunks = np.asarray(data[:n], dtype=np.int64).reshape(-1, block + 1)[:64]
    x = torch.from_numpy(chunks[:, :-1]).to(device)
    y = torch.from_numpy(chunks[:, 1:]).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).reshape(x.size(0), -1).mean(0)
    edges = torch.linspace(0, loss.numel(), bins + 1).long()
    return [loss[edges[i]:edges[i+1]].mean().item() for i in range(bins)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--gsm8k", type=int, default=200)
    p.add_argument("--val_npy", default=None, help="a val shard .npy to probe ppl-by-position")
    p.add_argument("--block", type=int, default=16384)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tok = Tokenizer.from_file(args.tokenizer)
    if args.gsm8k > 0:
        print(f"GSM8K exact-match ({args.gsm8k}): {gsm8k_em(model, tok, device, args.gsm8k):.3f}")
    if args.val_npy:
        bins = ppl_by_position(model, tok, device, args.val_npy, args.block)
        print("mean loss by position bin (early -> late):")
        print("  " + "  ".join(f"{b:.3f}" for b in bins))
        print("  (late bins should be <= early bins if long context is being used)")

if __name__ == "__main__":
    main()
```

- [ ] **Sanity check it parses:** `.venv/Scripts/python.exe -c "import ast; ast.parse(open('scripts/eval_longctx.py').read()); print('ok')"` → `ok`

### Task 0.8 — Commit and push

- [ ] Commit the fix + new files:

```bash
git checkout -b v3-longcontext
git add src/model.py configs/v3_*.yaml scripts/eval_longctx.py docs/runpod_longcontext_plan.md
git commit -m "v3: fix RoPE, add configurable rope_theta, long-context configs + eval"
git push -u origin v3-longcontext
```

---

## RunPod setup (one-time per pod)

- [ ] Rent a pod with a **persistent volume mounted at `/workspace`** (so checkpoints survive restarts). Use a **community/spot** GPU — the trainer saves on SIGTERM and auto-resumes, so preemption is safe and cheap. Start with a **24 GB** GPU (e.g., RTX 4090) for the 2K stages; switch to **48 GB** (A6000 / L40S / A40) for Phase 4 (16K).
- [ ] In the pod terminal:

```bash
cd /workspace
git clone -b v3-longcontext <your-repo-url> sambhav
cd sambhav
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login   # needed for gated/streamed datasets
```

- [ ] Confirm the GPU is visible: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- [ ] **Always run training under `nohup` + `--logs`** so a dropped SSH session doesn't kill the run, e.g. `nohup python train.py --config ... --resume auto --logs &` then `tail -f out/<run>/logs/*.log`.

---

## Phase 1 — Warm-start heal + base pretrain (2K, +4B tokens)

- [ ] **Make sure your existing 1B checkpoint is on the pod** at `out/runpod_sambhav_80m_v2_hybrid_1b/final.pt` (upload from local or mount your RunPod volume). `v3_base_2k.yaml` loads it via `train.base_checkpoint`; training errors out if it's missing.
- [ ] Prepare FineWeb-Edu tokens (only if missing):

```bash
python scripts/prepare_fineweb.py \
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --split train \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/fineweb_16k_5b \
  --max_tokens 5000000000 --val_tokens 5000000 --shard_tokens 50000000
```

- [ ] Inspect size/budget before committing GPU-hours: `python train.py --config configs/v3_base_2k.yaml --info`
- [ ] Train (warm-starts from your 1B automatically): `nohup python train.py --config configs/v3_base_2k.yaml --resume auto --logs &`
- [ ] Watch the log. Expected: an **initial loss bump** as attention re-adapts to the corrected RoPE, then loss settling *below* the old 1B level. Stop criterion: reaches `final.pt` at +4B tokens. If the loss never recovers below the old level after ~500M tokens, that's the signal warm-start didn't take — fall back to a fresh run (remove `base_checkpoint`).
- [ ] Smoke-test generation: `python generate.py --checkpoint out/v3_base_2k/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --prompt "Cloud computing is" --max_new_tokens 80`

## Phase 2 — Math CPT (2K, ~1B tokens)

- [ ] Prepare math tokens:

```bash
python scripts/prepare_lm_hf.py \
  --dataset_name open-web-math/open-web-math --split train \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/lm/openwebmath_2k \
  --format plain --text_field text \
  --max_tokens 1000000000 --val_tokens 2000000 --shard_tokens 50000000 --streaming true
```

- [ ] Train: `nohup python train.py --config configs/v3_cpt_math_2k.yaml --resume auto --logs &` (loads weights from `out/v3_base_2k/final.pt`).

## Phase 3 — Web CPT (2K, ~1B tokens)

- [ ] Prepare high-score Ultra-FineWeb tokens:

```bash
python scripts/prepare_lm_hf.py \
  --dataset_name openbmb/Ultra-FineWeb --dataset_config en --split train \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/lm/ultrafineweb_2k \
  --format plain --text_field content --min_score 0.9 \
  --max_tokens 1000000000 --val_tokens 2000000 --shard_tokens 50000000 --streaming true
```

- [ ] Train: `nohup python train.py --config configs/v3_cpt_web_2k.yaml --resume auto --logs &`

## Phase 4 — Context extension (16K, ~500M tokens) — switch to a 48 GB GPU

- [ ] Prepare long-document tokens from PG19 (books — naturally long, so a 16K window stays within one document):

```bash
python scripts/prepare_lm_hf.py \
  --dataset_name pg19 --split train \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/lm/pg19_16k \
  --format plain --text_field text \
  --max_tokens 500000000 --val_tokens 2000000 --shard_tokens 25000000 --streaming true
```

> If `pg19` 404s, use `--dataset_name deepmind/pg19`. Alternative long-doc sources: `togethercomputer/RedPajama-Data-1T-Sample` (arxiv/book subsets).

- [ ] `python train.py --config configs/v3_ctx16k.yaml --info` — confirm the auto-batch finder will likely land on `micro_batch=1`, `grad_accum≈8`.
- [ ] Train: `nohup python train.py --config configs/v3_ctx16k.yaml --resume auto --logs &`
- [ ] **If it OOMs at micro_batch=1:** lower `block_size` to `8192` (and halve the four window values), or jump to Phase 8 (FlexAttention) for a memory-efficient path.
- [ ] Validate long context worked:

```bash
python scripts/eval_longctx.py --checkpoint out/v3_ctx16k/final.pt \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json --gsm8k 0 \
  --val_npy data/lm/pg19_16k/val/shard_000000.npy --block 16384
```

Expected: later position bins have **≤** loss of earlier bins (model is using the long context).

## Phase 5 — Chat SFT (base = 16K checkpoint)

- [ ] Prepare UltraChat at 4K:

```bash
python scripts/prepare_sft.py \
  --dataset_name HuggingFaceH4/ultrachat_200k --split train_sft \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/sft/ultrachat_4k --max_seq_len 4096 \
  --val_fraction 0.02 --streaming true --overwrite
```

- [ ] Train: `nohup python train_sft.py --config configs/v3_sft_chat.yaml --resume auto --logs &`

## Phase 6 — Reasoning-polish SFT

- [ ] Prepare the reasoning mix (smoltalk + GSM8K by default; optional extra reasoning dataset via flags):

```bash
python scripts/prepare_reasoning_mix.py \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --out_dir data/sft/reasoning_polish_1024 --max_seq_len 1024 \
  --train_examples 50000 --val_examples 2000 --overwrite
```

- [ ] Train: `nohup python train_sft.py --config configs/v3_sft_reasoning.yaml --resume auto --logs &`

## Phase 7 — Final eval & download

- [ ] GSM8K + position eval on the final model:

```bash
python scripts/eval_longctx.py --checkpoint out/v3_sft_reasoning/best.pt \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json --gsm8k 200
```

- [ ] Instruction smoke test:

```bash
python generate_instruct.py --checkpoint out/v3_sft_reasoning/best.pt \
  --tokenizer tokenizer_fineweb_16k/tokenizer.json \
  --instruction "Explain why the sky is blue." --max_new_tokens 160
```

- [ ] Download the keepers off the pod before terminating: `out/v3_ctx16k/final.pt`, `out/v3_sft_chat/best.pt`, `out/v3_sft_reasoning/best.pt` (e.g. `runpodctl send` or `scp`).

---

## Phase 8 (OPTIONAL) — Unlock 32K–128K with FlexAttention

Only needed if 16K isn't enough. This replaces the O(T²) masks (see Risks) with `torch.nn.attention.flex_attention` block masks (O(N) memory, available in torch 2.11).

- [ ] In `src/model.py`, add a FlexAttention path for `sliding`/`csa`/`hca` using a `BlockMask` from a `mask_mod` (sliding window + block-summary visibility) instead of the boolean `attn_mask`. Keep the current masked path as a CPU/fallback.
- [ ] Re-run the Task 0.1 verification and a short 32K smoke run (`--max-steps 20`) to confirm no OOM and finite loss before committing to a full extension run at 32K (`rope_theta: 1000000`).

---

## Cost & time (rough — read real `tok/s` from logs)

| Phase | GPU | Tokens/steps | Ballpark wall-clock |
|---|---|---|---|
| 1 Base | 24 GB | 5B | 10–20 h |
| 2 Math CPT | 24 GB | 1B | 3–5 h |
| 3 Web CPT | 24 GB | 1B | 3–5 h |
| 4 Ctx 16K | 48 GB | 500M | 6–14 h |
| 5 Chat SFT | 24–48 GB | 1800 steps | 1–2 h |
| 6 Reasoning SFT | 24–48 GB | 500 steps | <1 h |

Total ≈ 25–45 GPU-hours. On community 4090/L40S pricing that's roughly **$15–45**. These are order-of-magnitude; the auto-batch finder + bf16 + your token throughput determine the real numbers.

---

## Risks & mitigations

1. **O(T²) attention masks at long context (biggest risk).** `sliding` builds a `[T,T]` bool mask (`model.py:106`) and `csa`/`hca` build `[T, T+blocks]` masks (`model.py:147-152`). At 16K that's ~0.27 GB *per masked layer call*; at 32K ~1 GB+ — only the `full` layers use memory-efficient FlashAttention. Mitigation in this plan: target 16K on a 48 GB GPU. Real fix for 32K+: Phase 8 (FlexAttention).
2. **No document-boundary masking** in `PackedMemmapDataset` — windows cross document boundaries. Mitigation: Phase 4 uses PG19 (long books), so a 16K window is mostly within one document. (A proper fix is intra-document attention masking, out of scope here.)
3. **Long-context retention through SFT.** SFT data is short, which can shrink effective context. Mitigation: keep model `block_size=16384` during SFT and use a 4K SFT length. If `eval_longctx` shows late-position loss rising after SFT, add some long-context SFT examples.
4. **Spot/community preemption.** Mitigation: persistent `/workspace` volume + `--resume auto` (already wired) + `save_on_interrupt` — a preempted run resumes from `last.pt`/`interrupted.pt` automatically on restart.
5. **Hybrid attention quality at 2K.** Always-on compressed layers may cost quality on short context. Mitigation: run `configs/ablate_2k_full.yaml` vs `configs/ablate_2k_hybrid.yaml` before committing the main 2K budget.
6. **Hybrid `torch.compile` is partial.** Full attention compiles cleanly, while `csa`/`hca` still use dynamic masks/block summaries and intentionally run eager under `torch.compile`. The opt-in FlexAttention path covers `sliding` only, and the code falls back to eager sliding unless the fused Triton-compiled FlexAttention path is available.
7. **Dataset IDs/fields drift on HuggingFace.** If a `--text_field` is wrong, the prep script prints available fields and exits — re-run with the right field. Confirm `open-web-math`, `openbmb/Ultra-FineWeb`, and `pg19` access on first use.

---

## Self-review notes

- Covers: RoPE correctness, rope_theta scaling, base over-training, two-phase 2K→16K context, domain CPT, instruction tuning, eval, and the 32K+ path. ✅
- Every config is full (no "same as above" code omissions for the structurally distinct files; the near-duplicates 0.4/0.6 list exact field diffs). ✅
- `base_checkpoint`/`out_dir` chain is consistent: `v3_base_2k → v3_cpt_math_2k → v3_cpt_web_2k → v3_ctx16k → v3_sft_chat → v3_sft_reasoning`. ✅
- Open decisions you may want to change before starting: target context (16K), base token budget (5B), dataset choices for math/web, and whether the 2K ablation favors full or hybrid attention.
