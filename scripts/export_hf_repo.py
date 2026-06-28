from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


RUNTIME_FILES = (
    "src/__init__.py",
    "src/model.py",
    "src/guided.py",
    "src/json_guided.py",
    "src/inference.py",
)


def repo_slug(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def slim_checkpoint(src: Path, dst: Path) -> dict:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ckpt["model"], "config": ckpt["config"]}, dst)
    return ckpt


def build_model_card(repo_id: str, ckpt: dict, checkpoint_in_repo: str, tokenizer_in_repo: str) -> str:
    model_cfg = dict(ckpt.get("config", {}).get("model", {}))
    project_name = ckpt.get("config", {}).get("project_name", repo_slug(repo_id))
    context = model_cfg.get("block_size", "unknown")
    layers = model_cfg.get("n_layer", "unknown")
    embd = model_cfg.get("n_embd", "unknown")
    heads = model_cfg.get("n_head", "unknown")
    kv_heads = model_cfg.get("n_kv_head", "unknown")
    attention = model_cfg.get("attention_mode", "unknown")
    pattern = model_cfg.get("hybrid_attention_pattern", "n/a")
    rope_theta = model_cfg.get("rope_theta", "unknown")
    vocab_size = ckpt["model"]["tok_embeddings.weight"].shape[0]
    param_count = sum(t.numel() for t in ckpt["model"].values())

    return f"""---
license: other
language:
- en
pipeline_tag: text-generation
library_name: pytorch
tags:
- pytorch
- language-model
- structured-output
- guided-decoding
- custom-code
---

# {repo_slug(repo_id)}

Custom PyTorch checkpoint for **{project_name}**, a compact Velora decoder-only language model trained for instruction and structured-output generation.

This is not a Hugging Face Transformers-format model. Use the included `generator.py` and local `src/` runtime files.

## Files

- `{checkpoint_in_repo}`: slim inference checkpoint containing `model` weights and `config`.
- `{tokenizer_in_repo}`: Tokenizers BPE tokenizer JSON.
- `generator.py`: CLI inference entry point for JSON/XML structured output.
- `src/`: minimal custom PyTorch model and guided-decoding runtime.
- `requirements.txt`: minimal inference dependencies.
- `config.json`: exported metadata for this bundle.

## Model Details

- Parameters: ~{param_count / 1_000_000:.1f}M
- Context length: {context}
- Vocabulary size: {vocab_size}
- Layers: {layers}
- Embedding dim: {embd}
- Attention heads / KV heads: {heads} / {kv_heads}
- Attention mode: {attention}
- Hybrid pattern: `{pattern}`
- RoPE theta: {rope_theta}

## Install

```bash
pip install -r requirements.txt
```

Install PyTorch for your CUDA/CPU environment if the default wheel is not correct for your machine.

## Usage

JSON-guided output:

```bash
python generator.py --checkpoint {checkpoint_in_repo} --tokenizer {tokenizer_in_repo} --format json --instruction "Return a JSON object with keys answer and confidence. Answer whether 2+2=4." --json_keys answer,confidence --json_key_types answer:boolean,confidence:number --temperature 0
```

Regex-guided output:

```bash
python generator.py --checkpoint {checkpoint_in_repo} --tokenizer {tokenizer_in_repo} --format json --instruction "Answer yes or no: is 2+2=4?" --regex "\\s*([Yy]es|[Nn]o)" --temperature 0
```

## Limitations

This is a small custom research/development checkpoint. The JSON and regex modes constrain decoding with local automata/parser-state logits masking, but model quality still depends on the checkpoint.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a self-contained Hugging Face model repo folder.")
    parser.add_argument("--repo_id", default="sambhav24/velora-100m-structured-strict-ctx16k")
    parser.add_argument("--checkpoint", default="out/sft_100m_structured_strict_ctx16k/best.pt")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--output", default=None, help="default: hf_repo/<repo-slug>")
    parser.add_argument("--checkpoint_in_repo", default="checkpoints/model.pt")
    parser.add_argument("--tokenizer_in_repo", default="tokenizer/tokenizer.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer = Path(args.tokenizer)
    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}")
    if not tokenizer.is_file():
        sys.exit(f"Tokenizer not found: {tokenizer}")

    out_dir = Path(args.output) if args.output else Path("hf_repo") / repo_slug(args.repo_id)
    if out_dir.exists():
        if not args.overwrite:
            sys.exit(f"Output already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ckpt = slim_checkpoint(checkpoint, out_dir / args.checkpoint_in_repo)
    copy_file(tokenizer, out_dir / args.tokenizer_in_repo)
    copy_file(Path("generate_structured.py"), out_dir / "generator.py")
    for rel in RUNTIME_FILES:
        copy_file(Path(rel), out_dir / rel)

    write_text(out_dir / "requirements.txt", "torch>=2.4\ntokenizers>=0.15\n")
    write_text(
        out_dir / "config.json",
        json.dumps(
            {
                "repo_id": args.repo_id,
                "checkpoint": args.checkpoint_in_repo,
                "tokenizer": args.tokenizer_in_repo,
                "source_checkpoint": str(checkpoint),
                "source_tokenizer": str(tokenizer),
                "model_config": ckpt.get("config", {}).get("model", {}),
            },
            indent=2,
        )
        + "\n",
    )
    write_text(out_dir / "README.md", build_model_card(args.repo_id, ckpt, args.checkpoint_in_repo, args.tokenizer_in_repo))

    print(f"Exported Hugging Face repo bundle: {out_dir}")
    print("Clean stale remote files first:")
    print(f"  .\\.venv\\Scripts\\hf.exe repos delete-files {args.repo_id} best.pt generate_structured.py velora_structured_strict_ctx16k_artifacts.tar src/")
    print("Then upload:")
    print(f"  .\\.venv\\Scripts\\hf.exe upload {args.repo_id} {out_dir} .")

if __name__ == "__main__":
    main()
