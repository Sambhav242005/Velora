from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


@dataclass
class TokenShard:
    path: Path
    data: np.ndarray

    @property
    def tokens(self) -> int:
        return int(self.data.shape[0])


class PackedMemmapDataset:
    def __init__(
        self,
        train_bin: Optional[str] = None,
        val_bin: Optional[str] = None,
        meta_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        seed: int = 1337,
    ):
        if data_dir is not None:
            root = Path(data_dir)
            meta_path = str(root / "meta.json")
        elif train_bin is None or val_bin is None or meta_path is None:
            raise ValueError("Provide either data_dir or train_bin, val_bin, and meta_path.")

        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.train_bin = Path(train_bin) if train_bin is not None else None
        self.val_bin = Path(val_bin) if val_bin is not None else None
        self.meta_path = Path(str(meta_path))

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {self.meta_path}")
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        dtype_name = self.meta.get("dtype", "uint16")
        if dtype_name not in DTYPE_MAP:
            raise ValueError(f"Unsupported token dtype in meta.json: {dtype_name}")
        self.dtype = DTYPE_MAP[dtype_name]
        self.train_tokens = int(self.meta["train_tokens"])
        self.val_tokens = int(self.meta["val_tokens"])
        self.vocab_size = int(self.meta["vocab_size"])

        self.train_shards = self._open_split("train")
        self.val_shards = self._open_split("val")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def _open_split(self, split: str) -> List[TokenShard]:
        if self.data_dir is not None:
            shard_dir = self.data_dir / split
            npy_shards = sorted(shard_dir.glob("*.npy")) if shard_dir.exists() else []
            if npy_shards:
                return [TokenShard(path=p, data=np.load(p, mmap_mode="r")) for p in npy_shards]

            bin_path = self.data_dir / f"{split}.bin"
            if bin_path.exists():
                tokens = self.train_tokens if split == "train" else self.val_tokens
                data = np.memmap(bin_path, dtype=self.dtype, mode="r", shape=(tokens,))
                return [TokenShard(path=bin_path, data=data)]

        bin_path = self.train_bin if split == "train" else self.val_bin
        if bin_path is None:
            raise FileNotFoundError(f"No {split} shards found under {self.data_dir}")
        if not bin_path.exists():
            raise FileNotFoundError(f"{split} token file not found: {bin_path}")
        tokens = self.train_tokens if split == "train" else self.val_tokens
        data = np.memmap(bin_path, dtype=self.dtype, mode="r", shape=(tokens,))
        return [TokenShard(path=bin_path, data=data)]

    def __len__(self) -> int:
        return self.train_tokens

    def get_rng_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.cpu())

    def get_batch(self, split: str, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        shards = self.train_shards if split == "train" else self.val_shards
        n = self.train_tokens if split == "train" else self.val_tokens
        if n < block_size + 1:
            raise ValueError(f"{split} dataset is too small: {n} tokens for block_size={block_size}")

        windows = np.asarray([shard.tokens - block_size for shard in shards], dtype=np.int64)
        valid = windows > 0
        if not np.any(valid):
            sizes = ", ".join(f"{s.path.name}:{s.tokens}" for s in shards)
            raise ValueError(
                f"No {split} shard has enough tokens for block_size={block_size}. "
                f"Shard sizes: {sizes}"
            )
        valid_shards = [shard for shard, ok in zip(shards, valid) if ok]
        valid_windows = windows[valid]
        cumulative = np.cumsum(valid_windows)
        total_windows = int(cumulative[-1])
        ix = torch.randint(0, total_windows, (batch_size,), generator=self.generator)

        x_list = []
        y_list = []
        for sample_idx in ix.tolist():
            shard_idx = int(np.searchsorted(cumulative, sample_idx, side="right"))
            prev = 0 if shard_idx == 0 else int(cumulative[shard_idx - 1])
            offset = int(sample_idx - prev)
            data = valid_shards[shard_idx].data
            chunk = np.asarray(data[offset:offset + block_size + 1], dtype=np.int64)
            x_list.append(torch.from_numpy(chunk[:-1]))
            y_list.append(torch.from_numpy(chunk[1:]))
        x = torch.stack(x_list).to(device=device, non_blocking=True)
        y = torch.stack(y_list).to(device=device, non_blocking=True)
        return x, y
