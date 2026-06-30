---
contentKind: case-study
title: Velora Custom LM Trainer
slug: velora
summary: A from-scratch PyTorch trainer for a compact LLaMA-style language model with resumable training and guided structured generation.
status: published
order: 1
featured: true
tags:
  - Python
  - PyTorch
  - Hugging Face
---

## Problem

Velora started as a practical constraint: train and publish a small language model from scratch without hiding the core system behind DeepSpeed, Hugging Face `Trainer`, or an opaque training stack. The project needed to run on a local Windows GPU and later on interruptible RunPod hardware, where failed runs, spot preemption, and memory limits are normal operating conditions.

The goal was not to compete with large hosted assistants. It was to build a debuggable end-to-end model pipeline: pretraining, continued pretraining, supervised fine-tuning, inference, evaluation, and export in a form that can actually be reproduced.

## Approach

The repository is organized around a config-driven training pipeline. YAML files define the model, data, optimizer, batch behavior, checkpointing, and evaluation settings for each run. Training data is prepared into memmapped token shards for language modeling and fixed-length padded arrays for instruction tuning, keeping data loading simple and inspectable.

The model is a compact LLaMA-style decoder implemented directly in PyTorch, with grouped-query attention, RoPE, RMSNorm, SwiGLU, tied embeddings, and hybrid attention modes for long-context experiments. The training path supports base pretraining, continued pretraining, and supervised fine-tuning, while inference scripts cover raw completion, instruction following, chat, regex-constrained output, and JSON-guided generation.

## Technical Decisions

The trainer stays single-process PyTorch on purpose. That keeps checkpoint behavior, memory recovery, and schedule logic visible in normal Python code, which matters more for this project than scaling across a cluster.

Resume safety is treated as a first-class feature. Checkpoints are written atomically and include model weights, optimizer state, scaler state, training state, RNG state, and data RNG state. The trainer can auto-discover resume candidates, save on interruption, and distinguish full training resume from warm-starting weights into a fresh optimizer schedule.

Long-context support is staged rather than assumed. The project uses short-context training first, then a context-extension phase with adjusted RoPE settings. Hybrid attention keeps a final global layer pattern for recovery, while the notes call out the remaining memory limit from explicit masks and the need for FlexAttention-style paths before pushing much beyond 16K.

Structured generation is implemented in the runtime instead of relying on prompt discipline alone. Regex automata and parser-state JSON masking guide logits during decoding, giving the small model a stricter path for yes/no and JSON-shaped outputs.

## Result

The project now has a runnable training and inference stack, published model-bundle workflow, Hugging Face export script, RunPod runbook, training-log viewer, evaluation scripts, and portfolio cover asset. The public model bundle includes the custom runtime files required to reproduce inference rather than only a checkpoint blob.

The outcome is a research and portfolio-scale language model system: compact, honest about its 80M-parameter capability ceiling, and built to show the engineering behind safe training, checkpoint recovery, long-context experiments, and constrained structured generation.
