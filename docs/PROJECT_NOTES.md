# Sambhav 80M — Engineering Notes, Bugs & Decisions

**Date:** 2026-06-05
**Branch:** `v3-longcontext`
**Scope:** Findings from a full read of the codebase, every bug/problem identified, the fixes applied, and the decisions taken to turn the project into a corrected, long-context, chat-focused 80M model trained on RunPod.

> Step-by-step run commands live in [`docs/runpod_longcontext_plan.md`](runpod_longcontext_plan.md). This file is the *why* (findings, bugs, decisions); the plan is the *how* (runbook).

---

## Table of contents
1. [Project overview](#1-project-overview)
2. [Bugs & problems (the full register)](#2-bugs--problems-the-full-register)
3. [Fixes & changes applied](#3-fixes--changes-applied)
4. [Decisions & rationale](#4-decisions--rationale)
5. [Realistic capability expectations](#5-realistic-capability-expectations)
6. [Training pipeline (v3)](#6-training-pipeline-v3)
7. [Open items & recommendations](#7-open-items--recommendations)
8. [File inventory / changelog](#8-file-inventory--changelog)

---

## 1. Project overview

A from-scratch ~80M LLaMA-style decoder-only LM ("Velora" / "sambhav-80m"). Pure single-process PyTorch, built for safe training on constrained / interruptible hardware (local GPU first, then RunPod).

- **Model** (`src/model.py`): GQA, RoPE, RMSNorm (pre-norm), SwiGLU, tied embeddings, no biases; residual-scaled init. Custom **hybrid attention** cycling a final-global local/compressed/global pattern per layer (csa/hca = block-summary "compressed" attention for cheap long range).
- **Data** (`src/data.py`, `src/sft_data.py`): memmapped token shards for pretraining; fixed-length padded arrays with prompt-masked labels for SFT.
- **Training** (`src/trainer.py`, `train.py`, `train_sft.py`): auto micro-batch finder under a VRAM cap, OOM-resilient batch halving, atomic checkpointing, save-on-SIGTERM/exception, full RNG-state resume, cosine LR.
- **Curriculum (as found):** 1B-token base pretrain → math CPT → ultra-fineweb CPT; separately chat SFT (UltraChat/smoltalk) → reasoning-polish SFT.
- **Stated goal (this engagement):** finish the general 80M as a usable **chat assistant**, with **long context**, reusing the already-trained 1B base.

---

## 2. Bugs & problems (the full register)

Severity: **CRITICAL** (breaks correctness) · **HIGH** (blocks a stated goal) · **MEDIUM** (hurts quality) · **LOW** (hygiene/minor) · **INFO** (not a bug, noted for clarity).

| # | Issue | Severity | Location | Status |
|---|---|---|---|---|
| B1 | RoPE convention mismatch — positional encoding not relative | **CRITICAL** | `src/model.py` rotate_half / rope cache | ✅ Fixed & verified |
| B2 | O(T²) attention masks materialized for sliding/csa/hca | **HIGH** (long ctx) | `src/model.py:106`, `:147` | ⚠️ Open — capped at 16K; FlexAttention fix deferred |
| B3 | No document-boundary masking in packed dataset | **MEDIUM** | `src/data.py` get_batch | ⚠️ Open — mitigated via long-doc data (PG19) |
| B4 | Base model undertrained (1B tok for 80M) | **MEDIUM** | training budget | ✅ Addressed (+4B warm-start) |
| B5 | ~~`.venv` committed into the repo~~ — **misdiagnosis** | LOW | `.gitignore` | ✅ Not an issue (already gitignored) |
| B6 | Tokenizer (16k, general text) poor for code | **LOW** (conditional) | `tokenizer_fineweb_16k` | ℹ️ Noted (only matters if doing code) |
| B7 | Always-on csa/hca compression may cost quality | **LOW/uncertain** | hybrid attention | ⬜ Ablation suggested, not run |
| B8 | Loss target remap `-100 → -1` then `ignore_index=-1` | **INFO** | `src/model.py:280` | ℹ️ Works; no action |
| B9 | `train.base_checkpoint` declared but ignored by LM/CPT trainer | **HIGH** | `src/trainer.py` init/resume path | ✅ Fixed & verified |
| B10 | Compiled checkpoints save `_orig_mod.`-prefixed model keys | **HIGH** | `src/trainer.py` save/resume path | ✅ Fixed |
| B11 | `torch.compile` fails on hybrid masked attention helpers | **MEDIUM** | `src/model.py` `_sliding_attention` / `_compressed_attention` | ✅ Mitigated |
| B12 | Hybrid `torch.compile` recompile thrash on `layer_idx` → eager fallback | **MEDIUM** | `src/model.py` `CausalSelfAttention.forward` / `_attention_kind` | ✅ Fixed |
| B13 | Gradient checkpointing was the hybrid 2K throughput ceiling (~1.8× tok/s recovered) | **PERF** | `configs/ablate_2k_hybrid.yaml` | ✅ Fixed |

### B1 — RoPE convention mismatch (CRITICAL) ✅ FIXED
**What:** `rotate_half` used the **interleaved** (even/odd) convention, while the `cos`/`sin` cache was built with `torch.cat((freqs, freqs))`, the **half-split** convention. Mixing the two means each (even, odd) dimension pair is rotated by *two different frequencies* — so it is not a rotation, and `q·k` no longer depends only on relative position `(m − n)`. RoPE's defining property was broken.

**Where:** `src/model.py` — `rotate_half` (was lines 43–46) and `_rope_cache` (`cat((freqs, freqs))`, line 79).

**Evidence (numerical, on the real code):** built a fixed `(q,k)`, applied RoPE across positions, and measured the spread of `⟨rope(q,m), rope(k,n)⟩` along each `m−n` diagonal (should be ~0 for correct RoPE):

| | max within-diagonal spread |
|---|---|
| Before (broken) | **4.99** |
| After fix, θ=10000 | **1.0e-05** |
| After fix, θ=500000 | **5.7e-06** |

**Fix:** `rotate_half` switched to the half-split form that matches the cache:
```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)
```

**Consequence:** the fix changes positional encoding, so **all checkpoints trained before it are incompatible** (the 1B base and its downstream math-CPT / smoltalk-SFT / reasoning-SFT). Handled via the warm-start decision (D1).

### B2 — O(T²) attention masks at long context (HIGH for long-context goal) ⚠️ OPEN
**What:** Only `full` attention uses memory-efficient FlashAttention (`is_causal=True`, no explicit mask). The other three modes build dense boolean masks and pass them to SDPA, which forces the O(T²) path:
- `sliding`: `_local_mask` builds a `[T, T]` bool (`src/model.py:106`).
- `csa`/`hca`: `_compressed_attention` builds a `[T, T + n_blocks]` bool (`src/model.py:147`).

**Impact:** ~0.27 GB per masked layer-call at 16K, ~1 GB+ at 32K — the very modes meant to make long context *cheap* are the ones that blow up memory. This caps how far context can scale.

**Mitigation (taken):** target **16K** on a 48 GB GPU (tolerable). **Real fix (deferred):** rewrite the masked paths with `torch.nn.attention.flex_attention` BlockMask (O(N) memory) — see Phase 8 of the plan. Required before 32K+.

### B3 — No document-boundary masking (MEDIUM) ⚠️ OPEN
**What:** `PackedMemmapDataset.get_batch` samples contiguous `block_size+1` windows from tokens packed across many documents (separated only by EOS). There is no intra-document attention mask, so at long context the model attends across unrelated documents.

**Impact:** at long context the model learns long *positions* but not necessarily long-range *dependencies*; cross-document attention is noise.

**Mitigation (taken):** the context-extension phase uses **PG19** (long books), so a 16K window mostly stays within one document. Proper fix (intra-doc masking) is out of scope for now.

### B4 — Base model undertrained (MEDIUM) ✅ ADDRESSED
**What:** 1B tokens for an 80M model ≈ **12 tokens/param** — below Chinchilla-optimal (~20) and far below the small-LM norm (SmolLM-135M: ~4,400; TinyLlama-1.1B: ~2,700). This is the biggest single quality limiter, more than architecture.

**Fix:** warm-start + **+4B** more clean tokens (≈5B effective). See D1.

### B5 — `.venv` committed into the repo — CORRECTED: not an issue ✅
**Original concern (incorrect):** that the virtual environment was tracked in git.
**Re-checked 2026-06-05:** `.gitignore` already excludes `.venv/`, `out/`, `data/`, `*.pt`/`*.pth`/`*.safetensors`/`*.ckpt`, `*.npy`/`*.bin`, `__pycache__/`, and logs. `git ls-files` reports **0** tracked files under `.venv/`, `out/`, or `data/`. No action needed — the earlier finding was wrong.

### B6 — Tokenizer not suitable for code (LOW, conditional) ℹ️ NOTED
**What:** `tokenizer_fineweb_16k` is a 16k BPE trained on web text. It tokenizes code inefficiently (whitespace/indentation, identifiers, symbols → many tokens), wasting context and capacity.
**Relevance:** only matters if the project pivots to code generation; then retrain a code-inclusive tokenizer and pretrain on code. Not relevant to the current chat goal.

### B7 — Always-on csa/hca compression may cost quality (LOW/uncertain) ⬜ UNVERIFIED
**What:** 3 of every 4 layers permanently use compressed attention. At `block_size=2048` this saves little compute but permanently reduces context fidelity in those layers — capacity a tiny model can ill afford. The Lighthouse paper's finding ("match dense only after a dense recovery phase") is indirect support that always-on compression can cost quality.
**Suggested test (not yet run):** train ~200–300M tokens with `attention_mode: full` vs `hybrid` and compare val loss. If `full` wins at 2K, drop the hybrid for the short-context stages and keep it only where long context is the point.

### B8 — Loss target remap (INFO, not a bug) ℹ️
`src/model.py:280` remaps label `-100 → -1` then calls `cross_entropy(..., ignore_index=-1)`. Works correctly (padding/prompt positions are ignored); just an unusual choice vs. using `ignore_index=-100` directly. No action.

### B9 — `train.base_checkpoint` ignored by LM/CPT trainer (HIGH) ✅ FIXED
**What:** The `v3_*` LM/CPT configs chain stages with `train.base_checkpoint`, but `Trainer` only tried same-folder resume checkpoints. A fresh `v3_base_2k` or CPT run would silently start from random initialization instead of loading the intended base weights.

**Fix:** `src/trainer.py` now treats resume as highest priority; if no full resume checkpoint is loaded, it loads `train.base_checkpoint` weights only and starts a fresh optimizer/schedule/RNG. `train.py --info` now reports whether a config will resume, warm-start, or start from scratch.

### B10 — `torch.compile` checkpoint key prefix (HIGH) ✅ FIXED
**What:** When `train.compile: true` wrapped the model, checkpoint saves used the compiled module's `state_dict()`, which prefixes every model key with `_orig_mod.`. Resume builds the plain model before compiling, so `load_state_dict` failed with all normal keys missing and all `_orig_mod.*` keys unexpected.

**Fix:** `src/trainer.py` now saves `self.model._orig_mod` when present and strips `_orig_mod.` while loading resume or warm-start checkpoints. This keeps compiled and non-compiled checkpoints interchangeable.

### B11 — `torch.compile` fails on hybrid masked attention helpers (MEDIUM) ✅ MITIGATED
**What:** `torch.compile` / TorchInductor failed on the hybrid `sliding`/`csa`/`hca` paths because their explicit mask and block-summary logic creates symbolic shape expressions that Inductor could not compare.

**Fix:** `src/model.py` now wraps the masked helper paths with `torch.compiler.disable` / `torch._dynamo.disable` when available. Full-attention layers and surrounding projections/FFNs remain compile-eligible, while the dynamic masked attention helpers run eager until they are rewritten with FlexAttention.

### B12 — Hybrid `torch.compile` recompile thrash → eager fallback (MEDIUM) ✅ FIXED
**What:** Hybrid runs (`ablate_2k_hybrid_compile`) showed ~no tok/s gain vs eager and logged `torch._dynamo hit config.recompile_limit (8)` with `last reason: self.layer_idx == 8`. `CausalSelfAttention.forward` called `_attention_kind()`, which indexes `pattern[self.layer_idx % len(pattern)]`. All layers share one `forward` code object, so Dynamo specialized it per `layer_idx` value (one graph per layer). With ≥8 layers this exceeded `recompile_limit` and the whole `forward` frame — **including the q/k/v/o projections**, not just the masked SDPA — fell back to eager. That, on top of B11's intentional helper graph-breaks, is why compile gave no speedup.

**Fix:** `src/model.py` resolves the attention kind once in `__init__` (`self._kind = self._attention_kind()`); `forward` reads the cached `self._kind` and no longer calls `_attention_kind()` or reads `self.layer_idx`, so the production guard `self.layer_idx == N` is impossible post-fix. Dynamo now guards on the kind string (≤4 distinct values) instead of `layer_idx` (one per layer).

**Verification (CPU, `backend="eager"`, real n_layer=24 hybrid structure):**
- Numerics: all 24 layers' `_kind` matches `pattern[i % 8]`; forward+backward give finite loss/grad-norm both with and without gradient checkpointing.
- Recompile: whole-model `torch.compile` with `recompile_limit=8, fail_on_recompile_limit_hit=True` — the **old** `layer_idx` branching raises `FailOnRecompileLimitHit` and reproduces the exact production warning (`self.layer_idx == 8 # mode = pattern[self.layer_idx % len(pattern)]`); the **fixed** version completes with no limit hit.
- `self._kind` is a plain str attribute → absent from `state_dict`; `load_state_dict(strict=True)` round-trips (resume/B10 unaffected).
- **GPU outcome (runpod, torch 2.4.1):** the `recompile_limit (8)` / `self.layer_idx == 8` warning is **gone** and numerics are bit-identical (loss 3.7320, val 3.5924). But steady-state tok/s did **not** move (~28k both ways) — see B13 for why: with the masked helpers still eager (B11) compile has only the elementwise glue to fuse, and the real ceiling was gradient checkpointing, not the recompile. The fix is still correct and worth keeping (removes wasted recompilation + eager fallback of projections); it just isn't a throughput lever on its own. The masked helpers remain `disable`d (B11) pending the FlexAttention rewrite (blocked: `flex_attention` needs torch ≥2.5; runpod has 2.4.1).

### B13 — Gradient checkpointing was the hybrid throughput ceiling at 2K (PERF) ✅ FIXED
**What:** `ablate_2k_hybrid*` ran `use_gradient_checkpointing: true`, which re-runs every block's forward during backward (an extra full forward per step, ~+50–100% compute). At 2K context this is pure overhead — the model has ~16 GB to spare without it. It also masked the real cost behind "compile does nothing" (B12): compile/eager/checkpointed all sat at ~28k tok/s because the second forward dominated.

**Evidence (runpod, 40-step smoke, hybrid+compile, GPU 44.4 GB):**

| config | micro_batch | reserved VRAM | steady tok/s |
|---|---|---|---|
| grad-ckpt ON (baseline) | 72 | 39.8 GB | ~28,000 |
| grad-ckpt OFF, auto mb=32 | 32→16 | 43.9 GB (edge) | 50,782 then OOM→recompile→20k |
| **grad-ckpt OFF, mb=20 (shipped)** | 20 | **27.7 GB** | **~50,400 (stable)** |

**Fix:** `configs/ablate_2k_hybrid.yaml` → `use_gradient_checkpointing: false`, `max_micro_batch: 20`, `max_vram_fraction: 0.85`. ~**1.8× tok/s**, stable (16 GB headroom), mathematically identical training (checkpointing only trades compute for memory). **Scope:** 2K only — re-enable checkpointing for ≥8K context where the activations no longer fit. Secondary finding: the auto-batch-finder underestimates real training memory by ~4 GB (compile workspace/fragmentation), so without checkpointing it picked an unstable mb=32 that OOM'd mid-run and triggered a torch.compile shape-recompile; capping `max_micro_batch` avoids it.

> **Process note (not a code bug):** the external paper at `arxiv.org/pdf/2605.06554` ("Lighthouse Attention", Nous Research) was assessed and found **out of scope** — its speedups apply only at 256K–1M context on datacenter GPUs, whereas this project runs ≤16K on a single GPU. Bookmark for a future long-context push, not now.

---

## 3. Fixes & changes applied

All on branch `v3-longcontext` (not yet committed at time of writing):

| Change | File(s) | Verified |
|---|---|---|
| RoPE rotate_half → half-split | `src/model.py` | ✅ diagonal-spread test (B1) |
| Added configurable `rope_theta` (default 10000) | `src/model.py` ModelConfig + rope cache | ✅ builds at θ=10000 & 500000 |
| 6 new training configs (`v3_*`) | `configs/v3_*.yaml` | ✅ all build 80.6M model |
| Warm-start base config from the 1B | `src/trainer.py`, `configs/v3_base_2k.yaml` | ✅ 1B loads `strict=True`; trainer loads weights when no resume exists |
| Compile-safe checkpoint save/load | `src/trainer.py`, `src/checkpoint.py`, `src/memory.py` | ✅ `py_compile`; RunPod failure mode reproduced from logs |
| Hybrid compile graph-break mitigation | `src/model.py` | ✅ Dynamo eager-backend compile smoke |
| 2K full-vs-hybrid ablation configs | `configs/ablate_2k_full*.yaml`, `configs/ablate_2k_hybrid*.yaml` | ✅ 50M + 200M configs build 80.6M model |
| Long-context eval (GSM8K + loss-by-position) | `scripts/eval_longctx.py` | ✅ parses/imports |
| Interactive multi-turn chat REPL | `chat.py` | ✅ parses/imports |
| RunPod runbook | `docs/runpod_longcontext_plan.md` | — |

**Re-verified 2026-06-05 (local):** all 6 `v3_*` configs build an 80.6M model; forward+backward is finite through every attention mode (`full`, `sliding`, `csa`, `hca`) and the gradient-checkpointing path; `generate()` runs; the 1B checkpoint warm-loads `strict=True`; tokenizer present.

---

## 4. Decisions & rationale

**D1 — Warm-start from the 1B, keep the RoPE fix.** The 1B base was trained with broken RoPE (B1), so it can't be cleanly resumed under the fixed code. Chosen path: load the 1B weights as **initialization** for the corrected model and continue training. Most learned features (embeddings, FFN, knowledge) transfer; only attention re-heals. Reuses the prior work while ending with a correct, long-context-capable model. Expect an initial **loss bump** during the heal; if it doesn't drop below the old 1B level within ~500M tokens, fall back to a fresh run. *(Alternatives rejected: "continue as-is / revert the fix" — keeps the bug and kills long context; "fresh retrain" — best quality but the 1B only cost a few GPU-hours, and warm-start should beat cold start anyway.)*

**D2 — Stage the positional changes.** Heal at `rope_theta=10000` (the value the 1B was trained at) for the 2K stages, so stage 1 only recovers from the rotation fix and not also a 50× frequency change. Raise to `rope_theta=500000` only at the 16K context-extension stage (the standard place to extend RoPE).

**D3 — Target 16K context (not 32K+).** 16K is a large jump from 2K, fits a 48 GB GPU, and avoids the O(T²) mask blow-up (B2). 32K+ is gated on the FlexAttention rewrite (plan Phase 8).

**D4 — Long-context via two-phase + long docs.** Pretrain at 2K (cheap), then a short extension phase at 16K on PG19 books (mitigates B3). Domain CPT (math, web) happens at 2K before extension.

**D5 — Chat-first framing, realistic scope.** Goal is a usable chat assistant as a learning/portfolio piece. Accepts the 80M ceiling (Section 5). The 16K stage is *optional* for chat (turns are short) — it can be skipped to ship faster by pointing `v3_sft_chat` at `v3_cpt_web_2k/final.pt`.

**D6 — Slightly gentler base LR (2e-4).** Matches the original 1B run so the warm-started features aren't blown away during the heal.

**D7 — Keep the hybrid pattern final-global.** Gemma-style local/global hybrids should end with a precise global path. The `v3_*` hybrid configs now use `full,sliding,csa,hca,sliding,csa,hca,full`, which keeps 6 of 24 layers global while making layer 23 `full` instead of `hca`.

**D8 — Ablate full vs hybrid at 2K before spending the main budget.** `configs/ablate_2k_full.yaml` and `configs/ablate_2k_hybrid.yaml` share data, token budget, optimizer, and warm-start checkpoint; the `*_compile_200m.yaml` variants extend the comparison to 200M sampled tokens. If full attention wins at 2K, use full for the short-context base/CPT stages and reserve hybrid for 16K.

---

## 5. Realistic capability expectations

An 80M model (~GPT-2-small class) trained on ~5–8B tokens is a **coherent, format-following chatbot**, not a reliable assistant. Fine-tuning sharpens and reformats capability; it does **not** add capability the scale can't support.

| Tier | Expectation |
|---|---|
| ✅ Will do | Fluent English; follows chat format; short simple instructions; common factual completions; basic autocomplete. |
| ⚠️ Hit or miss | Short summaries; simple Q&A; staying on-topic for a paragraph; *imitating* step-by-step reasoning. |
| ❌ Won't do reliably | Multi-step math (GSM8K ≈ 0), accurate facts (hallucinates), long coherent documents, complex/multi-constraint instructions, genuinely *using* 16K of context (it will *accept* 16K without collapsing, which is different from reasoning over it). |

**Domain specialization at 80M:** narrow + pattern-based tasks can be genuinely decent (code autocomplete of common patterns, one structured-generation task, single-step arithmetic); open-ended versions (write correct programs, solve novel math, reason over long docs) stay out of reach regardless of fine-tuning.

**The bigger lever, if "actually usable" matters most:** fine-tune an open small base (Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B, Gemma-3-1B) — still "your own model," but with real capability. The from-scratch 80M is the *learning* path; that knowledge transfers directly to fine-tuning.

---

## 6. Training pipeline (v3)

```
your 1B  (out/runpod_sambhav_80m_v2_hybrid_1b/final.pt)
   │  warm-start (load weights as initialization)
   ▼
v3_base_2k        2K, θ=10000, +4B tok, LR 2e-4     ← heals the RoPE fix
   ▼
v3_cpt_math_2k    2K, θ=10000                         (open-web-math)
   ▼
v3_cpt_web_2k     2K, θ=10000                         (Ultra-FineWeb, score-filtered)
   ▼
v3_ctx16k         16K, θ=500000                       ← context extension (PG19 books)
   ▼
v3_sft_chat       16K, chat SFT                       (UltraChat)
   ▼
v3_sft_reasoning  16K, reasoning polish               (smoltalk + GSM8K + reasoning)
   ▼
chat.py           ← talk to it
```

Full commands, dataset prep, RunPod setup, cost/time estimates, and risks: [`docs/runpod_longcontext_plan.md`](runpod_longcontext_plan.md).
Rough total: **~25–45 GPU-hours (~$15–45)** on community GPUs; the base heal+pretrain is the bulk.

---

## 7. Open items & recommendations

- [ ] **Commit + push** `v3-longcontext` (not yet committed).
- [ ] Put `out/runpod_sambhav_80m_v2_hybrid_1b/final.pt` on the RunPod persistent volume before Phase 1 (warm-start loads it).
- [ ] **B2 (FlexAttention)** — required before attempting 32K+; deferred (plan Phase 8).
- [ ] **B3 (doc masking)** — optional quality improvement for long context.
- [x] **B5 (.venv)** — not an issue; already gitignored (re-checked 2026-06-05).
- [ ] **B7 ablation** — run `configs/ablate_2k_full.yaml` vs `configs/ablate_2k_hybrid.yaml`.
- [x] **B9 warm-start loader** — fixed in `src/trainer.py`; verify with `train.py --info` before launching.
- [ ] Decide whether to keep the 16K stage (D5) or skip it for a faster chat model.
- [ ] Watch for the warm-start **loss bump** in Phase 1; fall back to fresh run if it doesn't recover (~500M tok).

---

## 8. File inventory / changelog

**Modified**
- `src/model.py` — RoPE fix (B1) + configurable `rope_theta`.
- `src/trainer.py` — LM/CPT `train.base_checkpoint` warm-start loader (B9).
- `train.py` — `--info` now reports resume vs base-checkpoint warm-start.
- `configs/v3_*.yaml` — hybrid pattern changed to end each 24-layer stack on a `full` layer.

**Created (configs)**
- `configs/ablate_2k_full.yaml` — 50M-token 2K full-attention comparison run.
- `configs/ablate_2k_hybrid.yaml` — matched 50M-token 2K hybrid comparison run.
- `configs/v3_base_2k.yaml` — warm-start heal + base pretrain (2K, θ=10000, +4B, base_checkpoint = 1B).
- `configs/v3_cpt_math_2k.yaml` — math CPT (2K, θ=10000).
- `configs/v3_cpt_web_2k.yaml` — web CPT (2K, θ=10000).
- `configs/v3_ctx16k.yaml` — context extension (16K, θ=500000).
- `configs/v3_sft_chat.yaml` — chat SFT (16K).
- `configs/v3_sft_reasoning.yaml` — reasoning-polish SFT (16K).

**Created (code/docs)**
- `scripts/eval_longctx.py` — GSM8K exact-match + loss-by-position eval.
- `scripts/prepare_lm_hf.py` — generic HF dataset to causal-LM token shards.
- `scripts/prepare_reasoning_mix.py` — smoltalk/GSM8K/optional reasoning SFT mix.
- `scripts/push_to_hf.py` — upload a checkpoint (+logs) to a HF model repo (reads `HF_TOKEN` from env; supports `--slim`).
- `chat.py` — interactive multi-turn chat REPL (matches SFT prompt format).
- `AGENTS.md` — agent working guide (commands, conventions, gotchas, how to update the project, secrets rule).
- `DEPLOYMENT.md` — secrets, artifact storage, inference, and serving.
- `.env.example` — template for `HF_TOKEN` / `WANDB_API_KEY` (real `.env` is gitignored).
- `docs/runpod_longcontext_plan.md` — the RunPod runbook.
- `docs/PROJECT_NOTES.md` — this document.

**Modified (config/hygiene)**
- `.gitignore` — added `.env` / `.env.*` (keeps `.env.example` tracked).

**Secrets:** no hardcoded keys anywhere (verified by grep). The only code that uses a key (`scripts/push_to_hf.py`) reads `HF_TOKEN` from the environment.
