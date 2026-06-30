---
contentKind: case-study
---

## Problem

Training language models fails in practical ways before it fails academically: datasets are too large, VRAM is tight, runs crash, and checkpoints get corrupted.

## Approach

I built Velora as a local smoke-test-first training stack before larger RunPod runs, with safe dataset preparation and restartable training loops.

## Technical Decisions

- LLaMA-style decoder-only transformer architecture with RoPE, RMSNorm, SwiGLU, and grouped-query attention.
- RAM-safe memmap loading for tokenized datasets.
- Automatic micro-batch search and VRAM logging to keep runs inside hardware limits.
- Atomic checkpoint writes and automatic resume from `last.pt` after interruption, OOM, crash, or SIGTERM.
- FineWeb-Edu preparation, tokenizer training, base training, and separate SFT workflows.

## Result

This is the strongest model-engineering proof in the portfolio because it shows infrastructure around training, not only notebook experimentation.
