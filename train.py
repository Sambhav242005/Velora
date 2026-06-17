from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from src.checkpoint import resume_checkpoint_candidates
from src.config import load_yaml
from src.data import PackedMemmapDataset
from src.model import GPT, ModelConfig
from src.trainer import Trainer


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_logs(out_dir: str, stem: str) -> Path:
    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    print(f"Logging to {log_path}")
    return log_path


def get_train_limit(cfg):
    train_cfg = cfg.get("train", {})
    max_steps = cfg.get("max_steps", train_cfg.get("max_steps"))
    max_tokens = cfg.get("max_tokens", train_cfg.get("max_tokens"))
    if max_steps is not None:
        return "steps", int(max_steps)
    if max_tokens is None:
        raise ValueError("Config must set either train.max_tokens or train.max_steps.")
    return "tokens", int(max_tokens)


def build_dataset(cfg):
    data_cfg = cfg.get("data", {})
    data_dir = cfg.get("data_dir") or data_cfg.get("data_dir")
    seed = int(cfg.get("seed", 1337))
    if data_dir:
        return PackedMemmapDataset(data_dir=data_dir, seed=seed)
    return PackedMemmapDataset(data_cfg["train_bin"], data_cfg["val_bin"], data_cfg["meta"], seed=seed)


def print_info(cfg, resume_override=None) -> None:
    dataset = build_dataset(cfg)
    model_cfg = dict(cfg["model"])
    if model_cfg.get("vocab_size") == "auto":
        model_cfg["vocab_size"] = dataset.vocab_size
    model_config = ModelConfig(**model_cfg)
    model = GPT(model_config)
    limit_kind, limit_value = get_train_limit(cfg)
    train_tokens = dataset.train_tokens
    val_tokens = dataset.val_tokens
    print(f"Out dir: {cfg['out_dir']}")
    print(f"Model params: {model.num_parameters()/1e6:.2f}M ({model.num_parameters():,})")
    print(
        "Model shape: "
        f"layers={model_config.n_layer}, embd={model_config.n_embd}, "
        f"heads={model_config.n_head}, kv_heads={model_config.n_kv_head}, block={model_config.block_size}"
    )
    print(
        "Attention: "
        f"mode={model_config.attention_mode}, pattern={model_config.hybrid_attention_pattern}, "
        f"sliding_window={model_config.sliding_window}, csa_block={model_config.csa_block_size}, "
        f"hca_block={model_config.hca_block_size}"
    )
    if model_config.dynamic_conv_qkv:
        print(
            "Dynamic conv: "
            f"qkv=true, kernel={model_config.dynamic_conv_kernel_size}, "
            f"rank={model_config.dynamic_conv_rank}, init_scale={model_config.dynamic_conv_init_scale}"
        )
    if model_config.recurrent_thinking:
        print(
            "Recurrent thinking: "
            f"steps={model_config.recurrent_thinking_steps}, "
            f"init_scale={model_config.recurrent_thinking_init_scale}"
        )
    print(f"Vocab size: {model_config.vocab_size:,}")
    print(f"Dataset tokens: train={train_tokens:,} | val={val_tokens:,} | total={train_tokens + val_tokens:,}")
    batch_cfg = cfg.get("batch", {})
    if batch_cfg.get("max_vram_gb") is not None:
        print(f"Batch VRAM cap: {float(batch_cfg['max_vram_gb']):.2f}GB reserved")
    elif batch_cfg.get("max_vram_fraction") is not None:
        print(f"Batch VRAM cap: {float(batch_cfg['max_vram_fraction']) * 100:.0f}% of GPU VRAM")
    if batch_cfg.get("max_micro_batch") is not None:
        print(f"Max micro-batch search: {int(batch_cfg['max_micro_batch'])}")
    if limit_kind == "steps":
        target_tokens = int(cfg["train"].get("target_tokens_per_update", 65536))
        approx_tokens = limit_value * target_tokens
        print(f"Training step budget: {limit_value:,} optimizer steps")
        print(f"Approx sampled tokens: {approx_tokens:,} (~{approx_tokens / max(1, train_tokens):.2f}x train split)")
    else:
        print(f"Training token budget: {limit_value:,} sampled tokens (~{limit_value / max(1, train_tokens):.2f}x train split)")

    resume_value = resume_override if resume_override is not None else cfg.get("checkpoint", {}).get("resume", "auto")
    candidates = resume_checkpoint_candidates(cfg["out_dir"], resume_value)
    if not candidates:
        print("Resume checkpoint: none")
        base_checkpoint = cfg.get("train", {}).get("base_checkpoint")
        if base_checkpoint:
            status = "exists" if Path(base_checkpoint).exists() else "missing"
            print(f"Base checkpoint warm-start: {base_checkpoint} ({status})")
        else:
            print("Base checkpoint warm-start: none")
        return
    ckpt_path = None
    ckpt = None
    for candidate in candidates:
        try:
            ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
            ckpt_path = candidate
            break
        except Exception as error:
            if resume_value != "auto":
                raise
            print(f"Skipping unreadable checkpoint {candidate}: {error}")
    if ckpt is None:
        print("Resume checkpoint: none readable")
        return
    state = ckpt.get("train_state", {})
    tokens_seen = int(state.get("tokens_seen", 0))
    steps_seen = int(state.get("step", 0))
    print(f"Resume checkpoint: {ckpt_path}")
    print(f"Checkpoint steps: {steps_seen:,}")
    print(f"Checkpoint tokens_seen: {tokens_seen:,}")
    if limit_kind == "steps":
        print(f"Remaining steps: {max(0, limit_value - steps_seen):,}")
    else:
        print(f"Remaining tokens: {max(0, limit_value - tokens_seen):,}")
    base_checkpoint = cfg.get("train", {}).get("base_checkpoint")
    if base_checkpoint:
        print("Base checkpoint warm-start: skipped because a resume checkpoint is available")


def main():
    parser = argparse.ArgumentParser(description="Safe local/RunPod LLM trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="auto, none, or path to checkpoint")
    parser.add_argument("--info", action="store_true", help="Print model/data/token budget info and exit")
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=None, help="Stop after this many optimizer steps; overrides train.max_tokens")
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=None, help="Override train.max_tokens when max_steps is not set")
    parser.add_argument("--max-vram-gb", "--max_vram_gb", dest="max_vram_gb", type=float, default=None, help="Pick the largest micro-batch under this peak reserved VRAM cap in GB")
    parser.add_argument("--max-vram-fraction", "--max_vram_fraction", dest="max_vram_fraction", type=float, default=None, help="Pick the largest micro-batch under this fraction of total GPU VRAM")
    parser.add_argument("--max-micro-batch", "--max_micro_batch", dest="max_micro_batch", type=int, default=None, help="Upper bound for auto micro-batch search")
    parser.add_argument("--auto-find-batch-on-resume", "--auto_find_batch_on_resume", dest="auto_find_batch_on_resume", action="store_true", help="Re-run the auto batch finder even when resuming")
    parser.add_argument("--logs", action="store_true", help="Tee stdout/stderr to out_dir/logs/train_*.log")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.logs:
        setup_logs(cfg["out_dir"], "train")
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps
    if args.max_tokens is not None:
        cfg.setdefault("train", {})["max_tokens"] = args.max_tokens
    if args.max_vram_gb is not None:
        cfg.setdefault("batch", {})["max_vram_gb"] = args.max_vram_gb
    if args.max_vram_fraction is not None:
        cfg.setdefault("batch", {})["max_vram_fraction"] = args.max_vram_fraction
        cfg.setdefault("batch", {})["auto_find_on_resume"] = True
    if args.max_micro_batch is not None:
        cfg.setdefault("batch", {})["max_micro_batch"] = args.max_micro_batch
        cfg.setdefault("batch", {})["auto_find_on_resume"] = True
    if args.auto_find_batch_on_resume:
        cfg.setdefault("batch", {})["auto_find_on_resume"] = True
    if args.info:
        print_info(cfg, resume_override=args.resume)
        return
    trainer = Trainer(cfg, resume_override=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
