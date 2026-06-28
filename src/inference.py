from __future__ import annotations

from pathlib import Path

_CHECKPOINT_NAMES = {"best.pt", "last.pt", "final.pt", "interrupted.pt", "emergency.pt"}


def checkpoint_error_message(path: str, root: str = "out", limit: int = 12) -> str:
    candidates = []
    out_root = Path(root)
    if out_root.exists():
        candidates = [
            str(candidate)
            for candidate in sorted(out_root.rglob("*.pt"))
            if candidate.name in _CHECKPOINT_NAMES or candidate.name.startswith("milestone_")
        ][:limit]
    if not candidates:
        return f"checkpoint not found: {path}. No checkpoints were found under {root}/."
    joined = "\n  ".join(candidates)
    return f"checkpoint not found: {path}. Available checkpoints:\n  {joined}"


def checkpoint_exists(path: str) -> bool:
    return Path(path).is_file()
