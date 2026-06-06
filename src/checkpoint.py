from __future__ import annotations

import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def atomic_torch_save(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def make_checkpoint(model, optimizer, scaler, train_state: Dict[str, Any], config: Dict[str, Any], data_rng_state=None) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_state": train_state,
        "config": config,
        "rng_state": rng_state(),
        "data_rng_state": data_rng_state,
    }


def rotate_milestones(out_dir: str | Path, keep_last_n: int = 3) -> None:
    out_dir = Path(out_dir)
    if keep_last_n <= 0:
        return
    checkpoints = sorted(out_dir.glob("milestone_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt in checkpoints[keep_last_n:]:
        try:
            ckpt.unlink()
        except OSError:
            pass


def resume_checkpoint_candidates(out_dir: str | Path, resume: str = "auto") -> list[Path]:
    out_dir = Path(out_dir)
    if resume in (None, "none", "false", False):
        return []
    if resume != "auto":
        p = Path(str(resume))
        return [p] if p.exists() else []

    candidates: list[Path] = []
    for name in ["last.pt", "interrupted.pt", "emergency.pt"]:
        p = out_dir / name
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(out_dir.glob("milestone_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    best = out_dir / "best.pt"
    if best.exists():
        candidates.append(best)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def find_resume_checkpoint(out_dir: str | Path, resume: str = "auto") -> Optional[Path]:
    candidates = resume_checkpoint_candidates(out_dir, resume)
    return candidates[0] if candidates else None
