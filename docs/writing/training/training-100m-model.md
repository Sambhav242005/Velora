---
contentKind: article
slug: training-100m-model
title: What I learned training a 100M custom LLM
type: technical-note
status: published
date: 2026-06-10
summary: Key takeaways from training the Velora-100M transformer model from scratch, including data prep and learning loops.
tags:
  - LLM
  - Training
  - PyTorch
---

Pretraining language models is usually the domain of large compute clusters. However, designing and pretraining a small 100M model locally provides invaluable insights into data pipelines, tokenization, and numerical stability. This post covers the design of Velora, a custom LLaMA-style decoder transformer built from scratch in PyTorch.

## Core Transformer Architecture

Velora does not use Hugging Face libraries; it is constructed from custom PyTorch layers:
- **Rotary Position Embeddings (RoPE)**: For better relative position modeling over standard absolute embeddings.
- **RMSNorm**: Replacing LayerNorm for faster computational speeds.
- **SwiGLU Activation**: Used in the feed-forward network to improve gradient flow.
- **Grouped Query Attention (GQA)**: To balance decoding speeds and key-value cache memory footprints.

## RAM-Safe Tokenization & Ingestion

Ingesting large datasets like FineWeb-Edu can easily crash local machine memory:
- **Custom Tokenizer**: Trained a byte-level BPE tokenizer (`tokenizer.json`).
- **Memory Mapping**: Utilized NumPy's `memmap` to stream tokenized binaries directly from disk, keeping RAM consumption under 2GB.

## Training Safety & Failure Recovery

Local training runs are prone to interruptions and Out-Of-Memory (OOM) errors. I added several safety measures to the training loop:
- **Micro-Batch Search**: The script executes a quick startup pass to identify the largest stable batch size for the local GPU's VRAM.
- **Atomic Checkpoint Writes**: Weights are written to a temporary file before replacing `last.pt`, ensuring that a crash mid-checkpoint does not corrupt the save.
