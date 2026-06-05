"""Upload a checkpoint (and optional logs) to a Hugging Face model repo.

The token is read from the HF_TOKEN environment variable — it is never taken
from code, config, or a command-line flag (so it can't leak into shell history
or logs). Create a write token at https://huggingface.co/settings/tokens and:

    Windows : [Environment]::SetEnvironmentVariable("HF_TOKEN","hf_xxx","User")
    Linux   : export HF_TOKEN=hf_xxx
    RunPod  : set HF_TOKEN as a pod Secret / environment variable

Examples:
    python scripts/push_to_hf.py --repo_id me/sambhav-80m --checkpoint out/v3_sft_chat/best.pt --slim
    python scripts/push_to_hf.py --repo_id me/sambhav-80m --checkpoint out/v3_ctx16k/final.pt --logs_dir out/v3_ctx16k/logs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def slim_checkpoint(src: Path, dst: Path) -> None:
    """Drop optimizer/scaler/rng state, keeping only what inference needs."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "config": ck["config"]}, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a checkpoint (+logs) to a Hugging Face model repo.")
    parser.add_argument("--repo_id", required=True, help="e.g. username/sambhav-80m")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt, e.g. out/v3_sft_chat/best.pt")
    parser.add_argument("--path_in_repo", default=None, help="destination path in the repo (default: checkpoints/<name>)")
    parser.add_argument("--logs_dir", default=None, help="optional logs folder to upload too")
    parser.add_argument("--slim", action="store_true", help="strip optimizer state before upload (inference-only, ~3x smaller)")
    parser.add_argument("--private", dest="private", action="store_true", default=True, help="create the repo private (default)")
    parser.add_argument("--public", dest="private", action="store_false", help="create the repo public")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens "
            "and expose it as the HF_TOKEN environment variable (never hardcode it)."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Missing dependency: `pip install -r requirements.txt` (huggingface_hub).")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    upload_path = checkpoint
    if args.slim:
        upload_path = checkpoint.with_suffix(".slim.pt")
        slim_checkpoint(checkpoint, upload_path)
        print(f"Wrote slim checkpoint: {upload_path}")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    dest = args.path_in_repo or f"checkpoints/{upload_path.name}"
    api.upload_file(
        path_or_fileobj=str(upload_path),
        path_in_repo=dest,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded {upload_path} -> {args.repo_id}:{dest}")

    if args.logs_dir:
        logs = Path(args.logs_dir)
        if logs.is_dir():
            in_repo = f"logs/{logs.parent.name}"
            api.upload_folder(
                folder_path=str(logs),
                path_in_repo=in_repo,
                repo_id=args.repo_id,
                repo_type="model",
            )
            print(f"Uploaded logs {logs} -> {args.repo_id}:{in_repo}")
        else:
            print(f"Logs dir not found, skipping: {logs}")


if __name__ == "__main__":
    main()
