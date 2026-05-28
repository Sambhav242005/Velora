from __future__ import annotations

import gc
from dataclasses import dataclass

import psutil
import torch


@dataclass
class MemoryStats:
    ram_used_gb: float
    ram_total_gb: float
    vram_allocated_gb: float
    vram_reserved_gb: float
    vram_free_gb: float
    vram_total_gb: float


def gb(x: int | float) -> float:
    return float(x) / (1024 ** 3)


class MemoryMonitor:
    def __init__(self, device: torch.device):
        self.device = device

    def stats(self) -> MemoryStats:
        vm = psutil.virtual_memory()
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            return MemoryStats(
                ram_used_gb=gb(vm.used), ram_total_gb=gb(vm.total),
                vram_allocated_gb=gb(allocated), vram_reserved_gb=gb(reserved),
                vram_free_gb=gb(free), vram_total_gb=gb(total),
            )
        return MemoryStats(
            ram_used_gb=gb(vm.used), ram_total_gb=gb(vm.total),
            vram_allocated_gb=0.0, vram_reserved_gb=0.0,
            vram_free_gb=0.0, vram_total_gb=0.0,
        )

    def short(self) -> str:
        s = self.stats()
        if self.device.type == "cuda":
            return (
                f"RAM {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB | "
                f"VRAM alloc {s.vram_allocated_gb:.1f}GB reserved {s.vram_reserved_gb:.1f}GB "
                f"free {s.vram_free_gb:.1f}/{s.vram_total_gb:.1f}GB"
            )
        return f"RAM {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB"


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
