from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


EOS_TOKEN_CANDIDATES = ("<eos>", "</s>", "<|endoftext|>", "[EOS]", "[SEP]")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


class ShardWriter:
    def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_tokens: int):
        self.split_dir = out_dir / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.shard_tokens = int(shard_tokens)
        self.buffer = np.empty((self.shard_tokens,), dtype=self.dtype)
        self.offset = 0
        self.shard_index = 0
        self.total_tokens = 0

    def write(self, token_ids: Sequence[int], limit: Optional[int] = None) -> int:
        remaining = len(token_ids) if limit is None else min(len(token_ids), limit)
        written = 0
        while remaining > 0:
            available = self.shard_tokens - self.offset
            n = min(available, remaining)
            start = written
            end = written + n
            self.buffer[self.offset:self.offset + n] = token_ids[start:end]
            self.offset += n
            self.total_tokens += n
            written += n
            remaining -= n
            if self.offset == self.shard_tokens:
                self.flush()
        return written

    def flush(self) -> None:
        if self.offset == 0:
            return
        path = self.split_dir / f"shard_{self.shard_index:06d}.npy"
        np.save(path, self.buffer[:self.offset])
        self.shard_index += 1
        self.offset = 0


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output folder already exists: {out_dir}. Pass --overwrite to replace FineWeb outputs.")
    if out_dir.exists() and overwrite:
        for child in (out_dir / "train", out_dir / "val"):
            if child.exists():
                shutil.rmtree(child)
        meta = out_dir / "meta.json"
        if meta.exists():
            meta.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def load_dataset_checked(dataset_name: str, dataset_config: str, split: str, streaming: bool):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets with `pip install -r requirements.txt`.") from exc

    try:
        return load_dataset(dataset_name, name=dataset_config, split=split, streaming=streaming)
    except Exception as exc:
        try:
            available = get_dataset_config_names(dataset_name)
            if dataset_config not in available:
                print(f"Dataset config {dataset_config!r} was not found for {dataset_name}.")
                print("Available configs:")
                for name in available:
                    print(f"  - {name}")
        except Exception as config_exc:
            print(f"Could not fetch available configs for {dataset_name}: {config_exc}")
        raise RuntimeError(
            f"Could not load dataset={dataset_name!r}, config={dataset_config!r}, split={split!r}, streaming={streaming}."
        ) from exc


def find_eos_id(tokenizer: Tokenizer) -> int:
    for token in EOS_TOKEN_CANDIDATES:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return int(token_id)
    raise ValueError(f"No EOS token found. Tried: {', '.join(EOS_TOKEN_CANDIDATES)}")


def resolve_text_field(example: Dict[str, Any], requested: Optional[str]) -> str:
    if requested:
        if requested not in example:
            raise ValueError(f"Text field {requested!r} not found. Available fields: {sorted(example.keys())}")
        if not isinstance(example[requested], str):
            raise ValueError(f"Text field {requested!r} exists but is not a string.")
        return requested

    for candidate in ("text", "content", "document", "raw_content"):
        if isinstance(example.get(candidate), str):
            return candidate

    for key, value in example.items():
        if isinstance(value, str):
            return key

    raise ValueError(f"No text field found. Available fields: {sorted(example.keys())}")


def iter_examples(dataset: Iterable[Dict[str, Any]], first: Dict[str, Any]):
    yield first
    yield from dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FineWeb/FineWeb-Edu token shards for safe local training.")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="data/fineweb_processed")
    parser.add_argument("--max_tokens", type=positive_int, required=True)
    parser.add_argument("--val_tokens", type=positive_int, default=500_000)
    parser.add_argument("--shard_tokens", type=positive_int, default=1_000_000)
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--streaming", type=parse_bool, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    out_dir = Path(args.out_dir)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if args.max_tokens < 1_000:
        raise ValueError("--max_tokens is too small. Use at least 1000 tokens.")
    if args.shard_tokens < 1_000:
        raise ValueError("--shard_tokens is too small. Use at least 1000 tokens.")

    print("Preparing Hugging Face dataset")
    print(f"  dataset_name:   {args.dataset_name}")
    print(f"  dataset_config: {args.dataset_config}")
    print(f"  split:          {args.split}")
    print(f"  tokenizer:      {tokenizer_path}")
    print(f"  target tokens:  train={args.max_tokens:,}, val={args.val_tokens:,}")
    print(f"  streaming:      {args.streaming}")

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output folder already exists: {out_dir}. Pass --overwrite to replace FineWeb outputs.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = find_eos_id(tokenizer)
    dtype = np.uint16 if vocab_size <= 65_535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"

    dataset = load_dataset_checked(args.dataset_name, args.dataset_config, args.split, args.streaming)
    iterator = iter(dataset)
    try:
        first_example = next(iterator)
    except StopIteration as exc:
        raise ValueError("Dataset split is empty.") from exc

    text_field = resolve_text_field(first_example, args.text_field)
    print(f"  text_field:     {text_field}")
    print(f"  vocab_size:     {vocab_size:,}")
    print(f"  dtype:          {dtype_name}")
    print(f"  shard_tokens:   {args.shard_tokens:,}")

    prepare_output_dir(out_dir, overwrite=args.overwrite)
    train_writer = ShardWriter(out_dir, "train", dtype, args.shard_tokens)
    val_writer = ShardWriter(out_dir, "val", dtype, args.shard_tokens)
    target_total = args.max_tokens + args.val_tokens
    progress_every = 100_000 if target_total <= 10_000_000 else 1_000_000
    next_progress = progress_every
    documents = 0

    with tqdm(total=target_total, unit="tok", desc="Tokenizing", dynamic_ncols=True) as pbar:
        for example in iter_examples(iterator, first_example):
            if val_writer.total_tokens >= args.val_tokens and train_writer.total_tokens >= args.max_tokens:
                break

            text = example.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids:
                continue
            ids.append(eos_id)
            documents += 1

            cursor = 0
            written_this_doc = 0
            if val_writer.total_tokens < args.val_tokens:
                remaining_val = args.val_tokens - val_writer.total_tokens
                wrote = val_writer.write(ids[cursor:], limit=remaining_val)
                cursor += wrote
                written_this_doc += wrote

            if cursor < len(ids) and train_writer.total_tokens < args.max_tokens:
                remaining_train = args.max_tokens - train_writer.total_tokens
                wrote = train_writer.write(ids[cursor:], limit=remaining_train)
                written_this_doc += wrote

            if written_this_doc:
                pbar.update(written_this_doc)

            total_written = val_writer.total_tokens + train_writer.total_tokens
            if total_written >= next_progress:
                tqdm.write(
                    f"Progress: {total_written:,}/{target_total:,} tokens "
                    f"(val={val_writer.total_tokens:,}, train={train_writer.total_tokens:,}, docs={documents:,})"
                )
                while next_progress <= total_written:
                    next_progress += progress_every

            if val_writer.total_tokens >= args.val_tokens and train_writer.total_tokens >= args.max_tokens:
                break

    train_writer.flush()
    val_writer.flush()

    if val_writer.total_tokens == 0:
        raise ValueError("No validation tokens were written.")
    if train_writer.total_tokens == 0:
        raise ValueError("No training tokens were written.")

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": int(vocab_size),
        "dtype": dtype_name,
        "train_tokens": int(train_writer.total_tokens),
        "val_tokens": int(val_writer.total_tokens),
        "shard_tokens": int(args.shard_tokens),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
