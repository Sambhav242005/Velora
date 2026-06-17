from __future__ import annotations

import platform
import sys


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("torch: not installed")
        print("Install PyTorch first, for example:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("status: CUDA is not visible to PyTorch")
        print("Check `nvidia-smi`, the RunPod image, and the installed PyTorch CUDA wheel.")
        return 1

    print(f"cuda device count: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"tf32 matmul enabled: {torch.backends.cuda.matmul.allow_tf32}")

    for device_idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_idx)
        major, minor = torch.cuda.get_device_capability(device_idx)
        vram_gb = props.total_memory / 1024**3
        print(f"\ndevice {device_idx}: {props.name}")
        print(f"  capability: ({major}, {minor})")
        print(f"  vram GB: {vram_gb:.2f}")
        print(f"  multiprocessors: {props.multi_processor_count}")

        if major < 9:
            print("  recommended dtype: bf16")
            print("  fp8 for this trainer: no; use bf16 on Ampere/Ada-class GPUs here")
        else:
            print("  recommended dtype: bf16 unless a tested FP8 training path is added")
            print("  fp8 for this trainer: hardware may support it, but the repo does not enable FP8")

    print("\nstatus: GPU smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
