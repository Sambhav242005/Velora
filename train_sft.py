from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.checkpoint import atomic_torch_save, make_checkpoint, resume_checkpoint_candidates, rng_state, set_rng_state
from src.config import load_yaml
from src.memory import MemoryMonitor, cleanup_cuda, gb
from src.model import GPT, ModelConfig
from src.sft_data import SFTArrayDataset


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


@dataclass
class SFTState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    last_save_step: int = 0
    last_eval_step: int = 0


class StopSignal:
    def __init__(self):
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        print(f"\nStop signal received ({signum}). Will save after current step...")
        self.stop_requested = True


class SFTTrainer:
    def __init__(self, config: Dict[str, Any], resume_override: Optional[str] = None):
        self.cfg = config
        self.sft_cfg = config.get("sft", {})
        self.train_cfg = config.get("train", {})
        self.batch_cfg = config.get("batch", {})
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
        self.dataset = SFTArrayDataset(config["data"]["data_dir"], seed=seed)
        self.max_steps = int(self.sft_cfg.get("max_steps", self.train_cfg.get("max_steps", 1000)))
        if self.max_steps <= 0:
            raise ValueError("sft.max_steps must be positive.")

        self.dtype = self.resolve_dtype(self.train_cfg.get("dtype", "bf16"))
        self.use_amp = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        scaler_enabled = self.device.type == "cuda" and self.dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        resume_value = resume_override if resume_override is not None else config.get("checkpoint", {}).get("resume", "auto")
        resume_candidates = resume_checkpoint_candidates(self.out_dir, resume_value)
        resume_ckpt = self.load_first_readable_resume(resume_candidates, resume_value)
        if resume_ckpt is not None:
            self.model = self.model_from_checkpoint(resume_ckpt).to(self.device)
        else:
            base_checkpoint = self.sft_cfg.get("base_checkpoint")
            if not base_checkpoint:
                raise ValueError("sft.base_checkpoint is required when no SFT resume checkpoint exists.")
            print(f"Loading base checkpoint: {base_checkpoint}")
            base_ckpt = torch.load(base_checkpoint, map_location=self.device, weights_only=False)
            self.model = self.model_from_checkpoint(base_ckpt).to(self.device)

        self.optimizer = self.configure_optimizer()
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.last_grad_norm: Optional[float] = None
        self.state = SFTState()
        self.state.micro_batch_size = int(self.batch_cfg.get("start_micro_batch", 1))
        self.state.grad_accum_steps = self.compute_grad_accum(self.state.micro_batch_size)

        if resume_ckpt is not None:
            self.restore_training_state(resume_ckpt)

        vram_limit = self.vram_limit_gb()
        print(f"Device: {self.device}")
        print(f"Model params: {self.model.num_parameters()/1e6:.2f}M")
        print(f"SFT examples: train={self.dataset.train_examples:,} | val={self.dataset.val_examples:,}")
        print(f"Target SFT steps: {self.max_steps:,}")
        if vram_limit is not None:
            print(f"Batch VRAM cap: {vram_limit:.2f}GB peak reserved")
        print(f"Initial memory: {self.monitor.short()}")

        if self.batch_cfg.get("auto_find", True) and (self.state.step == 0 or self.batch_cfg.get("auto_find_on_resume", True)):
            self.auto_find_batch()

    def load_first_readable_resume(self, candidates, resume_value):
        if not candidates:
            return None
        for candidate in candidates:
            try:
                print(f"Trying resume checkpoint: {candidate}")
                ckpt = torch.load(candidate, map_location=self.device, weights_only=False)
                print(f"Resuming from {candidate}")
                return ckpt
            except Exception as error:
                if resume_value != "auto":
                    raise
                print(f"Skipping unreadable checkpoint {candidate}: {error}")
        return None

    def model_from_checkpoint(self, ckpt: Dict[str, Any]) -> GPT:
        model_cfg = dict(ckpt["config"]["model"])
        model_cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
        model = GPT(ModelConfig(**model_cfg))
        model.load_state_dict(ckpt["model"])
        return model

    def restore_training_state(self, ckpt: Dict[str, Any]) -> None:
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        ts = ckpt.get("train_state", {})
        self.state = SFTState(**{k: ts.get(k, getattr(SFTState(), k)) for k in SFTState.__dataclass_fields__.keys()})
        if ckpt.get("rng_state"):
            set_rng_state(ckpt["rng_state"])
        if ckpt.get("data_rng_state") is not None:
            self.dataset.set_rng_state(ckpt["data_rng_state"])
        self.state.grad_accum_steps = self.compute_grad_accum(self.state.micro_batch_size)
        print(
            f"Resumed step={self.state.step}, tokens_seen={self.state.tokens_seen}, "
            f"remaining_steps={max(0, self.max_steps - self.state.step):,}, micro_batch={self.state.micro_batch_size}"
        )

    def resolve_dtype(self, name: str):
        name = str(name).lower()
        if name == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if name == "bf16" and torch.cuda.is_available():
            return torch.float16
        if name == "fp16":
            return torch.float16
        return torch.float32

    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
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
        no_decay = (no_decay & param_dict.keys()) | (param_dict.keys() - decay)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": float(self.train_cfg.get("weight_decay", 0.0))},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=float(self.train_cfg.get("learning_rate", 5e-5)),
            betas=(float(self.train_cfg.get("beta1", 0.9)), float(self.train_cfg.get("beta2", 0.95))),
            fused=(self.device.type == "cuda"),
        )

    def target_tokens_per_update(self) -> int:
        return int(self.train_cfg.get("target_tokens_per_update", 32768))

    def compute_grad_accum(self, micro_batch: int) -> int:
        denom = max(1, micro_batch * self.dataset.max_seq_len)
        return max(1, math.ceil(self.target_tokens_per_update() / denom))

    def get_lr(self) -> float:
        lr = float(self.train_cfg.get("learning_rate", 5e-5))
        min_lr = float(self.train_cfg.get("min_lr", lr * 0.1))
        warmup = int(self.train_cfg.get("warmup_steps", 20))
        if self.state.step < warmup:
            return lr * (self.state.step + 1) / max(1, warmup)
        decay_ratio = (self.state.step - warmup) / max(1, self.max_steps - warmup)
        decay_ratio = min(1.0, max(0.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def grad_norm(self) -> float:
        total_sq = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            total_sq += float(param.grad.detach().float().pow(2).sum().cpu())
        return math.sqrt(total_sq)

    def write_metrics(self, record: Dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def vram_limit_gb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        if self.batch_cfg.get("max_vram_gb") is not None:
            return float(self.batch_cfg["max_vram_gb"])
        if self.batch_cfg.get("max_vram_fraction") is not None:
            return self.monitor.stats().vram_total_gb * float(self.batch_cfg["max_vram_fraction"])
        return None

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
            x, y = self.dataset.get_batch("train", micro_batch, self.device)
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

    def auto_find_batch(self) -> None:
        if self.device.type != "cuda":
            print("CPU mode: skipping auto batch finder.")
            return
        max_mb = int(self.batch_cfg.get("max_micro_batch", 64))
        limit = self.vram_limit_gb()
        print(f"Finding safe SFT micro-batch size under {limit:.2f}GB peak reserved VRAM..." if limit else "Finding safe SFT micro-batch size...")

        measurements: Dict[int, tuple[bool, float, bool]] = {}

        def check(micro_batch: int) -> tuple[bool, float, bool]:
            if micro_batch not in measurements:
                ok, peak_reserved_gb = self.profile_microbatch(micro_batch)
                within_limit = ok and (limit is None or peak_reserved_gb <= limit)
                measurements[micro_batch] = (ok, peak_reserved_gb, within_limit)
                if ok:
                    cap_text = f" / cap={limit:.2f}GB" if limit is not None else ""
                    status = "OK" if within_limit else "over cap"
                    print(f"  micro_batch={micro_batch}: peak_reserved={peak_reserved_gb:.2f}GB{cap_text} -> {status}")
            return measurements[micro_batch]

        best = None
        upper = max_mb
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
            print(f"Warning: micro_batch=1 is above the VRAM cap at {peak_reserved_gb:.2f}GB.")

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

    def save(self, name: str = "last.pt") -> None:
        path = self.out_dir / name
        obj = make_checkpoint(
            self.model,
            self.optimizer,
            self.scaler,
            asdict(self.state),
            self.cfg,
            data_rng_state=self.dataset.get_rng_state(),
        )
        atomic_torch_save(obj, path)
        if name != "last.pt":
            atomic_torch_save(obj, self.out_dir / "last.pt")
        print(f"Saved checkpoint: {path}")

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        mb = self.state.micro_batch_size
        accum = self.state.grad_accum_steps
        for _ in range(accum):
            x, y = self.dataset.get_batch("train", mb, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)
                loss = loss / accum
            total_loss += float(loss.detach().cpu())
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        self.last_grad_norm = self.grad_norm()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        lr = self.get_lr()
        self.set_lr(lr)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.state.step += 1
        self.state.tokens_seen += mb * accum * self.dataset.max_seq_len
        return total_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        eval_iters = int(self.cfg.get("eval", {}).get("iters", 50))
        eval_bs = int(self.cfg.get("eval", {}).get("batch_size", min(4, self.state.micro_batch_size)))
        for _ in range(eval_iters):
            x, y = self.dataset.get_batch("val", eval_bs, self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)
            losses.append(float(loss.cpu()))
        cleanup_cuda()
        return sum(losses) / len(losses)

    def train(self) -> None:
        print("Starting SFT...")
        print(f"Already trained steps: {self.state.step:,}")
        print(f"Remaining steps: {max(0, self.max_steps - self.state.step):,}")
        print(f"micro_batch={self.state.micro_batch_size}, grad_accum={self.state.grad_accum_steps}")
        log_interval = int(self.train_cfg.get("log_interval", 10))
        eval_interval = int(self.train_cfg.get("eval_interval_steps", 50))
        save_interval = int(self.train_cfg.get("save_interval_steps", 100))

        last_log_time = time.time()
        last_log_tokens = self.state.tokens_seen
        try:
            while self.state.step < self.max_steps:
                loss = self.train_step()

                if self.state.step % log_interval == 0:
                    now = time.time()
                    dt = max(1e-6, now - last_log_time)
                    tok_s = (self.state.tokens_seen - last_log_tokens) / dt
                    last_log_time = now
                    last_log_tokens = self.state.tokens_seen
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"step {self.state.step:,}/{self.max_steps:,} | loss {loss:.4f} | lr {lr:.2e} | "
                        f"mb {self.state.micro_batch_size} | accum {self.state.grad_accum_steps} | "
                        f"grad {self.last_grad_norm:.2f} | tok/s {tok_s:.0f} | {self.monitor.short()}"
                    )
                    mem = self.monitor.stats()
                    self.write_metrics({
                        "event": "train",
                        "step": self.state.step,
                        "tokens_seen": self.state.tokens_seen,
                        "loss": loss,
                        "lr": lr,
                        "micro_batch": self.state.micro_batch_size,
                        "grad_accum": self.state.grad_accum_steps,
                        "tokens_per_second": tok_s,
                        "grad_norm": self.last_grad_norm,
                        "ram_used_gb": mem.ram_used_gb,
                        "vram_allocated_gb": mem.vram_allocated_gb,
                        "vram_reserved_gb": mem.vram_reserved_gb,
                    })

                if self.state.step - self.state.last_eval_step >= eval_interval:
                    val_loss = self.evaluate()
                    self.state.last_eval_step = self.state.step
                    print(f"VAL | step {self.state.step:,} | val_loss {val_loss:.4f}")
                    self.write_metrics({
                        "event": "eval",
                        "step": self.state.step,
                        "tokens_seen": self.state.tokens_seen,
                        "val_loss": val_loss,
                    })
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self.save("best.pt")

                if self.state.step - self.state.last_save_step >= save_interval:
                    self.state.last_save_step = self.state.step
                    self.save(f"milestone_{self.state.step}.pt")

                if self.stopper.stop_requested:
                    print("Stop requested. Saving interrupted checkpoint...")
                    self.save("interrupted.pt")
                    return

            print("SFT target reached. Saving final checkpoint...")
            self.save("final.pt")
        finally:
            try:
                self.save("last.pt")
            except Exception as save_error:
                print(f"Final save failed: {save_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction fine-tune a base checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="auto, none, or path to checkpoint")
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--max-vram-gb", "--max_vram_gb", dest="max_vram_gb", type=float, default=None)
    parser.add_argument("--max-micro-batch", "--max_micro_batch", dest="max_micro_batch", type=int, default=None)
    parser.add_argument("--logs", action="store_true", help="Tee stdout/stderr to out_dir/logs/sft_*.log")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.logs:
        setup_logs(cfg["out_dir"], "sft")
    if args.max_steps is not None:
        cfg.setdefault("sft", {})["max_steps"] = args.max_steps
    if args.max_vram_gb is not None:
        cfg.setdefault("batch", {})["max_vram_gb"] = args.max_vram_gb
    if args.max_micro_batch is not None:
        cfg.setdefault("batch", {})["max_micro_batch"] = args.max_micro_batch
    trainer = SFTTrainer(cfg, resume_override=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
