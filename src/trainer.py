from __future__ import annotations

import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .checkpoint import atomic_torch_save, make_checkpoint, resume_checkpoint_candidates, rng_state, rotate_milestones, set_rng_state
from .data import PackedMemmapDataset
from .memory import MemoryMonitor, cleanup_cuda, gb
from .model import ModelConfig, GPT


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    last_save_tokens: int = 0
    last_eval_tokens: int = 0


class StopSignal:
    def __init__(self):
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print(f"\nStop signal received ({signum}). Will save after current step...")
        self.stop_requested = True


class Trainer:
    def __init__(self, config: Dict[str, Any], resume_override: Optional[str] = None):
        self.cfg = config
        self.out_dir = Path(config["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stopper = StopSignal()

        seed = int(config.get("seed", 1337))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = MemoryMonitor(self.device)

        data_cfg = config.get("data", {})
        data_dir = config.get("data_dir") or data_cfg.get("data_dir")
        if data_dir:
            self.dataset = PackedMemmapDataset(data_dir=data_dir, seed=seed)
        else:
            self.dataset = PackedMemmapDataset(
                data_cfg["train_bin"],
                data_cfg["val_bin"],
                data_cfg["meta"],
                seed=seed,
            )

        model_cfg = dict(config["model"])
        if model_cfg.get("vocab_size") == "auto":
            model_cfg["vocab_size"] = self.dataset.vocab_size
        self.model_cfg = ModelConfig(**model_cfg)
        self.model = GPT(self.model_cfg).to(self.device)

        self.state = TrainState()
        self.state.micro_batch_size = int(config["batch"].get("start_micro_batch", 1))
        self.state.grad_accum_steps = self.compute_grad_accum(self.state.micro_batch_size)
        self.max_tokens = self.resolve_optional_positive_int(config.get("max_tokens", config["train"].get("max_tokens")), "max_tokens")
        self.max_steps = self.resolve_optional_positive_int(config.get("max_steps", config["train"].get("max_steps")), "max_steps")
        if self.max_tokens is None and self.max_steps is None:
            raise ValueError("Config must set either train.max_tokens or train.max_steps.")

        self.dtype = self.resolve_dtype(config["train"].get("dtype", "bf16"))
        self.use_amp = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        scaler_enabled = self.device.type == "cuda" and self.dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.optimizer = self.configure_optimizer()

        self.start_time = time.time()
        self.last_save_time = time.time()

        print(f"Device: {self.device}")
        print(f"Model params: {self.model.num_parameters()/1e6:.2f}M")
        print(f"Vocab size: {self.model_cfg.vocab_size}")
        print(f"Block size: {self.model_cfg.block_size}")
        print(
            "Attention: "
            f"mode={self.model_cfg.attention_mode}, pattern={self.model_cfg.hybrid_attention_pattern}, "
            f"sliding_window={self.model_cfg.sliding_window}, csa_block={self.model_cfg.csa_block_size}, "
            f"hca_block={self.model_cfg.hca_block_size}"
        )
        if self.model_cfg.dynamic_conv_qkv:
            print(
                "Dynamic conv: "
                f"qkv=true, kernel={self.model_cfg.dynamic_conv_kernel_size}, "
                f"rank={self.model_cfg.dynamic_conv_rank}, init_scale={self.model_cfg.dynamic_conv_init_scale}"
            )
        print(
            f"Dataset tokens: train={self.dataset.train_tokens:,} | "
            f"val={self.dataset.val_tokens:,} | total={self.dataset.train_tokens + self.dataset.val_tokens:,}"
        )
        if self.max_steps is not None:
            approx_tokens = self.max_steps * self.target_tokens_per_update()
            print(f"Training step budget: {self.max_steps:,} optimizer steps (~{approx_tokens / max(1, self.dataset.train_tokens):.2f}x train split at target tokens/update)")
        elif self.dataset.train_tokens > 0:
            print(f"Training token budget: {self.max_tokens:,} sampled tokens (~{self.max_tokens / self.dataset.train_tokens:.2f}x train split)")
        vram_limit = self.vram_limit_gb()
        if vram_limit is not None:
            print(f"Batch VRAM cap: {vram_limit:.2f}GB peak reserved")
        print(f"Initial memory: {self.monitor.short()}")

        resume_value = resume_override if resume_override is not None else config.get("checkpoint", {}).get("resume", "auto")
        resumed = self.try_resume(resume_value)
        if not resumed:
            self.try_warm_start()

        if config["train"].get("compile", False) and hasattr(torch, "compile"):
            print("Compiling model with torch.compile... first step may be slow.")
            self.model = torch.compile(self.model)

        batch_cfg = config.get("batch", {})
        auto_find_on_resume = bool(batch_cfg.get("auto_find_on_resume", False) or batch_cfg.get("max_vram_gb") is not None)
        if batch_cfg.get("auto_find", True) and (self.state.step == 0 or auto_find_on_resume):
            self.auto_find_batch()

    def resolve_dtype(self, name: str):
        name = str(name).lower()
        if name == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if name == "bf16" and torch.cuda.is_available():
            return torch.float16
        if name == "fp16":
            return torch.float16
        return torch.float32

    def resolve_optional_positive_int(self, value: Any, name: str) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive when set.")
        return value

    def resolve_optional_positive_float(self, value: Any, name: str) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive when set.")
        return value

    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif "norm" in fpn:
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        decay = decay & param_dict.keys()
        no_decay = no_decay & param_dict.keys()
        no_decay = no_decay | (param_dict.keys() - decay)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": float(self.cfg["train"].get("weight_decay", 0.1))},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=float(self.cfg["train"].get("learning_rate", 3e-4)),
            betas=(float(self.cfg["train"].get("beta1", 0.9)), float(self.cfg["train"].get("beta2", 0.95))),
            fused=(self.device.type == "cuda"),
        )

    def compute_grad_accum(self, micro_batch: int) -> int:
        target = self.target_tokens_per_update()
        denom = max(1, micro_batch * self.model_cfg.block_size)
        return max(1, math.ceil(target / denom))

    def target_tokens_per_update(self) -> int:
        return int(self.cfg["train"].get("target_tokens_per_update", 65536))

    def vram_limit_gb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        batch_cfg = self.cfg.get("batch", {})
        max_vram_gb = self.resolve_optional_positive_float(batch_cfg.get("max_vram_gb"), "batch.max_vram_gb")
        if max_vram_gb is not None:
            return max_vram_gb
        max_vram_fraction = batch_cfg.get("max_vram_fraction")
        if max_vram_fraction is None:
            return None
        max_vram_fraction = float(max_vram_fraction)
        if max_vram_fraction <= 0 or max_vram_fraction > 1:
            raise ValueError("batch.max_vram_fraction must be > 0 and <= 1.")
        return self.monitor.stats().vram_total_gb * max_vram_fraction

    def get_lr(self) -> float:
        lr = float(self.cfg["train"].get("learning_rate", 3e-4))
        min_lr = float(self.cfg["train"].get("min_lr", lr * 0.1))
        warmup = int(self.cfg["train"].get("warmup_steps", 1000))
        if self.max_steps is not None:
            max_steps = self.max_steps
        else:
            max_steps = max(1, math.ceil(self.max_tokens / max(1, self.target_tokens_per_update())))
        if self.state.step < warmup:
            return lr * (self.state.step + 1) / max(1, warmup)
        if self.state.step >= max_steps:
            return min_lr
        decay_ratio = (self.state.step - warmup) / max(1, max_steps - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def checkpoint_model(self) -> torch.nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    def evaluation_model(self) -> torch.nn.Module:
        # Validation runs under no-grad/eval mode. Calling the compiled training
        # wrapper there makes Dynamo specialize again on grad_mode and can hit its
        # recompile limit; use the original module for infrequent eval passes.
        return self.checkpoint_model()

    def normalize_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefix = "_orig_mod."
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
        return state_dict

    def is_dynamic_conv_key(self, key: str) -> bool:
        return any(part in key for part in (".q_dyn_conv.", ".k_dyn_conv.", ".v_dyn_conv."))

    def load_warm_start_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = [
            key for key in missing
            if self.model_cfg.dynamic_conv_qkv and self.is_dynamic_conv_key(key)
        ]
        disallowed_missing = [key for key in missing if key not in allowed_missing]
        if disallowed_missing or unexpected:
            details = []
            if disallowed_missing:
                details.append(f"missing={disallowed_missing[:8]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:8]}")
            raise RuntimeError("Warm-start checkpoint is incompatible: " + "; ".join(details))
        if allowed_missing:
            print(
                "Warm-start checkpoint has no dynamic-conv weights; "
                f"initialized {len(allowed_missing)} new dynamic-conv tensors from config."
            )

    def try_resume(self, resume_value: str) -> bool:
        candidates = resume_checkpoint_candidates(self.out_dir, resume_value)
        if not candidates:
            return False
        ckpt_path = None
        ckpt = None
        for candidate in candidates:
            try:
                print(f"Trying resume checkpoint: {candidate}")
                ckpt = torch.load(candidate, map_location=self.device, weights_only=False)
                ckpt_path = candidate
                break
            except Exception as error:
                if resume_value != "auto":
                    raise
                print(f"Skipping unreadable checkpoint {candidate}: {error}")
        if ckpt is None:
            print("No readable auto-resume checkpoint found.")
            return False
        print(f"Resuming from {ckpt_path}")
        self.model.load_state_dict(self.normalize_checkpoint_state_dict(ckpt["model"]))
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        ts = ckpt.get("train_state", {})
        self.state = TrainState(**{k: ts.get(k, getattr(TrainState(), k)) for k in TrainState.__dataclass_fields__.keys()})
        if ckpt.get("rng_state"):
            set_rng_state(ckpt["rng_state"])
        if ckpt.get("data_rng_state") is not None:
            self.dataset.set_rng_state(ckpt["data_rng_state"])
        if self.max_steps is not None:
            remaining = max(0, self.max_steps - self.state.step)
            remaining_text = f"remaining_steps={remaining}"
        else:
            remaining = max(0, self.max_tokens - self.state.tokens_seen)
            remaining_text = f"remaining_tokens={remaining}"
        print(
            f"Resumed step={self.state.step}, tokens_seen={self.state.tokens_seen}, "
            f"{remaining_text}, micro_batch={self.state.micro_batch_size}"
        )
        return True

    def try_warm_start(self) -> bool:
        base_checkpoint = self.cfg.get("train", {}).get("base_checkpoint")
        if not base_checkpoint:
            return False
        path = Path(base_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"train.base_checkpoint not found: {path}")
        print(f"Warm-starting weights from base checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_warm_start_state_dict(self.normalize_checkpoint_state_dict(state_dict))
        print("Loaded base checkpoint weights only; optimizer, scheduler, RNG, and data order start fresh.")
        return True

    def save(self, name: str = "last.pt") -> None:
        path = self.out_dir / name
        obj = make_checkpoint(
            self.checkpoint_model(),
            self.optimizer,
            self.scaler,
            asdict(self.state),
            self.cfg,
            data_rng_state=self.dataset.get_rng_state(),
        )
        atomic_torch_save(obj, path)
        if name != "last.pt":
            atomic_torch_save(obj, self.out_dir / "last.pt")
        self.last_save_time = time.time()
        print(f"Saved checkpoint: {path}")

    def profile_microbatch(self, micro_batch: int) -> tuple[bool, float]:
        saved_rng_state = rng_state()
        saved_data_rng_state = self.dataset.get_rng_state()
        self.model.train()
        cleanup_cuda()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        x = y = logits = loss = None
        try:
            x, y = self.dataset.get_batch("train", micro_batch, self.model_cfg.block_size, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits, loss = self.model(x, y)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_reserved_gb = gb(torch.cuda.max_memory_reserved(self.device))
            else:
                peak_reserved_gb = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            return True, peak_reserved_gb
        except torch.cuda.OutOfMemoryError:
            print(f"OOM while testing micro_batch={micro_batch}")
            self.optimizer.zero_grad(set_to_none=True)
            return False, float("inf")
        finally:
            del x, y, logits, loss
            self.dataset.set_rng_state(saved_data_rng_state)
            set_rng_state(saved_rng_state)
            cleanup_cuda()

    def can_fit_microbatch(self, micro_batch: int, respect_vram_limit: bool = True) -> bool:
        ok, peak_reserved_gb = self.profile_microbatch(micro_batch)
        if not ok:
            return False
        limit = self.vram_limit_gb() if respect_vram_limit else None
        if limit is not None and peak_reserved_gb > limit:
            print(
                f"micro_batch={micro_batch} fits but exceeds VRAM cap: "
                f"peak_reserved={peak_reserved_gb:.2f}GB > cap={limit:.2f}GB"
            )
            return False
        return True

    def auto_find_batch(self) -> None:
        if self.device.type != "cuda":
            print("CPU mode: skipping auto batch finder.")
            return
        start = max(1, int(self.cfg["batch"].get("start_micro_batch", 1)))
        max_mb = int(self.cfg["batch"].get("max_micro_batch", 32))
        if max_mb < 1:
            raise ValueError("batch.max_micro_batch must be positive.")
        limit = self.vram_limit_gb()
        limit_text = f" under {limit:.2f}GB peak reserved VRAM" if limit is not None else ""
        print(f"Finding safe micro-batch size{limit_text}...")

        measurements: Dict[int, tuple[bool, float, bool]] = {}

        def check(micro_batch: int) -> tuple[bool, float, bool]:
            if micro_batch not in measurements:
                ok, peak_reserved_gb = self.profile_microbatch(micro_batch)
                within_limit = ok and (limit is None or peak_reserved_gb <= limit)
                measurements[micro_batch] = (ok, peak_reserved_gb, within_limit)
                if ok:
                    status = "OK" if within_limit else "over cap"
                    cap_text = f" / cap={limit:.2f}GB" if limit is not None else ""
                    print(f"  micro_batch={micro_batch}: peak_reserved={peak_reserved_gb:.2f}GB{cap_text} -> {status}")
            return measurements[micro_batch]

        best = None
        upper = max_mb
        mb = min(start, max_mb)
        if mb != 1:
            mb = 1
        while mb <= max_mb:
            ok, _, within_limit = check(mb)
            if not ok or not within_limit:
                upper = mb - 1
                break
            best = mb
            mb *= 2
        else:
            upper = max_mb

        if best is None:
            ok, peak_reserved_gb, _ = check(1)
            if not ok:
                raise RuntimeError("micro_batch=1 does not fit in VRAM.")
            best = 1
            print(
                f"Warning: micro_batch=1 needs {peak_reserved_gb:.2f}GB peak reserved VRAM, "
                f"which is above the configured cap {limit:.2f}GB. Using micro_batch=1."
            )

        low = best + 1
        high = upper
        while low <= high:
            mid = (low + high) // 2
            ok, _, within_limit = check(mid)
            if ok and within_limit:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        self.state.micro_batch_size = best
        self.state.grad_accum_steps = self.compute_grad_accum(best)
        cleanup_cuda()
        print(f"Auto batch selected: micro_batch={best}, grad_accum={self.state.grad_accum_steps}")

    def reduce_batch_after_oom(self) -> bool:
        old = self.state.micro_batch_size
        if old <= 1:
            print("OOM at micro_batch=1. Cannot reduce further. Try smaller block_size/model.")
            return False
        new = max(1, old // 2)
        self.state.micro_batch_size = new
        self.state.grad_accum_steps = self.compute_grad_accum(new)
        print(f"Reduced micro_batch {old} -> {new}, grad_accum={self.state.grad_accum_steps}")
        cleanup_cuda()
        return True

    def maybe_increase_batch(self) -> None:
        batch_cfg = self.cfg.get("batch", {})
        if not batch_cfg.get("increase_if_safe", False):
            return
        if self.device.type != "cuda":
            return
        every = int(batch_cfg.get("increase_every_steps", 200))
        if self.state.step == 0 or self.state.step % every != 0:
            return
        max_mb = int(batch_cfg.get("max_micro_batch", 64))
        candidate = self.state.micro_batch_size * 2
        if candidate > max_mb:
            return
        stats = self.monitor.stats()
        limit = self.vram_limit_gb()
        if limit is not None:
            has_headroom = stats.vram_reserved_gb < limit * 0.65
        else:
            used_fraction = stats.vram_reserved_gb / max(0.01, stats.vram_total_gb)
            max_fraction = float(batch_cfg.get("max_vram_fraction", 0.9))
            has_headroom = used_fraction < max_fraction * 0.65
        if has_headroom and self.can_fit_microbatch(candidate):
            old = self.state.micro_batch_size
            self.state.micro_batch_size = candidate
            self.state.grad_accum_steps = self.compute_grad_accum(candidate)
            print(f"Increased micro_batch {old} -> {candidate}, grad_accum={self.state.grad_accum_steps}")

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        mb = self.state.micro_batch_size
        accum = self.state.grad_accum_steps
        for _ in range(accum):
            x, y = self.dataset.get_batch("train", mb, self.model_cfg.block_size, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)
                loss = loss / accum
            total_loss += float(loss.detach().cpu())
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        if float(self.cfg["train"].get("grad_clip", 0.0)) > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["train"].get("grad_clip", 1.0)))
        lr = self.get_lr()
        self.set_lr(lr)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        tokens_this_step = mb * accum * self.model_cfg.block_size
        self.state.step += 1
        self.state.tokens_seen += tokens_this_step
        return total_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        eval_model = self.evaluation_model()
        eval_model.eval()
        losses = []
        eval_iters = int(self.cfg.get("eval", {}).get("iters", 20))
        eval_bs = int(self.cfg.get("eval", {}).get("batch_size", min(4, self.state.micro_batch_size)))
        for _ in range(eval_iters):
            x, y = self.dataset.get_batch("val", eval_bs, self.model_cfg.block_size, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                _, loss = eval_model(x, y)
            losses.append(float(loss.cpu()))
        cleanup_cuda()
        return sum(losses) / len(losses)

    def target_reached(self) -> bool:
        if self.max_steps is not None:
            return self.state.step >= self.max_steps
        return self.state.tokens_seen >= self.max_tokens

    def train(self) -> None:
        print("Starting training...")
        if self.max_steps is not None:
            print(f"Target steps: {self.max_steps:,}")
            print(f"Already trained steps: {self.state.step:,}")
            print(f"Remaining steps: {max(0, self.max_steps - self.state.step):,}")
            print(f"Already trained tokens: {self.state.tokens_seen:,}")
        else:
            print(f"Target tokens: {self.max_tokens:,}")
            print(f"Already trained tokens: {self.state.tokens_seen:,}")
            print(f"Remaining tokens: {max(0, self.max_tokens - self.state.tokens_seen):,}")
        print(f"micro_batch={self.state.micro_batch_size}, grad_accum={self.state.grad_accum_steps}")
        log_interval = int(self.cfg["train"].get("log_interval", 10))
        eval_interval_tokens = int(self.cfg["train"].get("eval_interval_tokens", self.cfg["train"].get("eval_every_tokens", 10_000_000)))
        save_interval_tokens = int(self.cfg["train"].get("save_interval_tokens", self.cfg["train"].get("save_every_tokens", 25_000_000)))
        save_interval_minutes = float(self.cfg["train"].get("save_interval_minutes", self.cfg["train"].get("save_every_minutes", 30)))
        keep_last_n = int(self.cfg.get("checkpoint", {}).get("keep_last_n", 3))

        last_log_time = time.time()
        last_log_tokens = self.state.tokens_seen
        try:
            while not self.target_reached():
                try:
                    loss = self.train_step()
                except torch.cuda.OutOfMemoryError:
                    print("CUDA OOM during training step. Saving emergency checkpoint...")
                    self.save("emergency.pt")
                    if not self.reduce_batch_after_oom():
                        raise
                    continue

                if self.state.step % log_interval == 0:
                    now = time.time()
                    dt = max(1e-6, now - last_log_time)
                    tok_s = (self.state.tokens_seen - last_log_tokens) / dt
                    last_log_time = now
                    last_log_tokens = self.state.tokens_seen
                    lr = self.optimizer.param_groups[0]["lr"]
                    if self.max_steps is not None:
                        progress = f"step {self.state.step:,}/{self.max_steps:,} | tokens {self.state.tokens_seen:,}"
                    else:
                        progress = f"step {self.state.step:,} | tokens {self.state.tokens_seen:,}/{self.max_tokens:,}"
                    print(
                        f"{progress} | loss {loss:.4f} | lr {lr:.2e} | mb {self.state.micro_batch_size} | "
                        f"accum {self.state.grad_accum_steps} | tok/s {tok_s:.0f} | {self.monitor.short()}"
                    )

                if self.state.tokens_seen - self.state.last_eval_tokens >= eval_interval_tokens:
                    val_loss = self.evaluate()
                    self.state.last_eval_tokens = self.state.tokens_seen
                    print(f"VAL | tokens {self.state.tokens_seen:,} | val_loss {val_loss:.4f}")
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self.save("best.pt")

                time_due = (time.time() - self.last_save_time) >= save_interval_minutes * 60
                tokens_due = (self.state.tokens_seen - self.state.last_save_tokens) >= save_interval_tokens
                if time_due or tokens_due:
                    self.state.last_save_tokens = self.state.tokens_seen
                    milestone = f"milestone_{self.state.tokens_seen}.pt"
                    self.save(milestone)
                    rotate_milestones(self.out_dir, keep_last_n=keep_last_n)

                self.maybe_increase_batch()

                if self.stopper.stop_requested:
                    print("Stop requested. Saving interrupted checkpoint...")
                    self.save("interrupted.pt")
                    return

            print("Training target reached. Saving final checkpoint...")
            self.save("final.pt")
        except Exception:
            if self.cfg.get("checkpoint", {}).get("save_on_exception", True):
                print("Exception happened. Saving crash checkpoint before re-raising...")
                try:
                    self.save("crash.pt")
                except Exception as save_error:
                    print(f"Failed to save crash checkpoint: {save_error}")
            raise
        finally:
            try:
                self.save("last.pt")
            except Exception as save_error:
                print(f"Final save failed: {save_error}")
