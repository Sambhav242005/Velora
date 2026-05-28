from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class SFTArrayDataset:
    def __init__(self, data_dir: str, seed: int = 1337):
        self.data_dir = Path(data_dir)
        self.meta_path = self.data_dir / "meta.json"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"SFT metadata not found: {self.meta_path}")
        self.meta: Dict = json.loads(self.meta_path.read_text(encoding="utf-8"))

        self.train_input_ids = np.load(self.data_dir / "train_input_ids.npy", mmap_mode="r")
        self.train_labels = np.load(self.data_dir / "train_labels.npy", mmap_mode="r")
        self.val_input_ids = np.load(self.data_dir / "val_input_ids.npy", mmap_mode="r")
        self.val_labels = np.load(self.data_dir / "val_labels.npy", mmap_mode="r")

        if self.train_input_ids.shape != self.train_labels.shape:
            raise ValueError("train_input_ids and train_labels shapes do not match.")
        if self.val_input_ids.shape != self.val_labels.shape:
            raise ValueError("val_input_ids and val_labels shapes do not match.")

        self.max_seq_len = int(self.meta["max_seq_len"])
        self.vocab_size = int(self.meta["vocab_size"])
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    @property
    def train_examples(self) -> int:
        return int(self.train_input_ids.shape[0])

    @property
    def val_examples(self) -> int:
        return int(self.val_input_ids.shape[0])

    def get_rng_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state.cpu())

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        if split == "train":
            input_ids = self.train_input_ids
            labels = self.train_labels
        elif split == "val":
            input_ids = self.val_input_ids
            labels = self.val_labels
        else:
            raise ValueError(f"Unsupported split: {split}")

        n = input_ids.shape[0]
        ix = torch.randint(0, n, (batch_size,), generator=self.generator).tolist()
        x = torch.from_numpy(np.asarray(input_ids[ix], dtype=np.int64)).to(device=device, non_blocking=True)
        y = torch.from_numpy(np.asarray(labels[ix], dtype=np.int64)).to(device=device, non_blocking=True)
        return x, y
