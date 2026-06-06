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
        raise FileExistsError(f"Output folder already exists: {out_dir}. Pass --overwrite to replace it.")
    if out_dir.exists() and overwrite:
        for child in (out_dir / "train", out_dir / "val"):
            if child.exists():
                shutil.rmtree(child)
        meta = out_dir / "meta.json"
        if meta.exists():
            meta.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def load_dataset_checked(dataset_name: str, dataset_config: Optional[str], split: str, streaming: bool):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets with `pip install -r requirements.txt`.") from exc

    kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        try:
            available = get_dataset_config_names(dataset_name)
            if dataset_config and dataset_config not in available:
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


def field_text(example: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = example.get(name)
        if isinstance(value, str):
            return value.strip()
    return ""


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


def message_role(message: Dict[str, Any]) -> str:
    role = str(message.get("role", message.get("from", ""))).strip().lower()
    if role in {"user", "human"}:
        return "User"
    if role in {"assistant", "gpt", "bot"}:
        return "Assistant"
    if role == "system":
        return "System"
    return role.title() if role else "Message"


def message_content(message: Dict[str, Any]) -> str:
    return str(message.get("content", message.get("value", ""))).strip()


def chat_text(example: Dict[str, Any], messages_field: str) -> str:
    messages = example.get(messages_field)
    if not isinstance(messages, list):
        return ""
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message_content(message)
        if content:
            lines.append(f"{message_role(message)}: {content}")
    return "\n".join(lines)


def extract_text(example: Dict[str, Any], args: argparse.Namespace, text_field: str) -> str:
    if args.min_score is not None:
        score = example.get(args.score_field)
        if not isinstance(score, (int, float)) or float(score) < args.min_score:
            return ""

    fmt = args.format
    if fmt == "auto":
        if isinstance(example.get(args.messages_field), list):
            fmt = "chat"
        elif args.text_field or any(isinstance(example.get(k), str) for k in ("text", "content", "document", "raw_content")):
            fmt = "plain"
        elif field_text(example, args.question_field) and field_text(example, args.answer_field):
            fmt = "qa"
        else:
            fmt = "input_output"

    if fmt == "plain":
        value = example.get(text_field)
        return value.strip() if isinstance(value, str) else ""
    if fmt == "qa":
        question = field_text(example, args.question_field, "question", "prompt", "instruction")
        answer = field_text(example, args.answer_field, "answer", "response", "output")
        return f"Question:\n{question}\n\nAnswer:\n{answer}" if question and answer else ""
    if fmt == "input_output":
        instruction = field_text(example, args.instruction_field, "instruction", "prompt", "question")
        context = field_text(example, args.input_field, "input", "context")
        output = field_text(example, args.output_field, "output", "response", "answer", "completion")
        if not instruction or not output:
            return ""
        if context:
            return f"Instruction:\n{instruction}\n\nInput:\n{context}\n\nResponse:\n{output}"
        return f"Instruction:\n{instruction}\n\nResponse:\n{output}"
    if fmt == "chat":
        return chat_text(example, args.messages_field)
    raise ValueError(f"Unsupported format: {args.format}")


def iter_examples(dataset: Iterable[Dict[str, Any]], first: Dict[str, Any]):
    yield first
    yield from dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare generic Hugging Face datasets into causal-LM token shards.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--format", choices=["auto", "plain", "qa", "input_output", "chat"], default="auto")
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--instruction_field", default="instruction")
    parser.add_argument("--input_field", default="input")
    parser.add_argument("--output_field", default="output")
    parser.add_argument("--messages_field", default="messages")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--max_tokens", type=positive_int, required=True)
    parser.add_argument("--val_tokens", type=positive_int, default=500_000)
    parser.add_argument("--shard_tokens", type=positive_int, default=1_000_000)
    parser.add_argument("--streaming", type=parse_bool, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_path = Path(args.tokenizer)
    out_dir = Path(args.out_dir)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if args.shard_tokens < 1_000:
        raise ValueError("--shard_tokens is too small. Use at least 1000 tokens.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = find_eos_id(tokenizer)
    dtype = np.uint16 if vocab_size <= 65_535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"

    dataset = load_dataset_checked(args.dataset_name, args.dataset_config, args.split, args.streaming)
    iterator = iter(dataset)
    try:
        first_example = dict(next(iterator))
    except StopIteration as exc:
        raise ValueError("Dataset split is empty.") from exc

    text_field = ""
    if args.format == "plain" or args.text_field:
        text_field = resolve_text_field(first_example, args.text_field)
    elif args.format == "auto" and not isinstance(first_example.get(args.messages_field), list):
        text_field = resolve_text_field(first_example, None)
    prepare_output_dir(out_dir, overwrite=args.overwrite)
    train_writer = ShardWriter(out_dir, "train", dtype, args.shard_tokens)
    val_writer = ShardWriter(out_dir, "val", dtype, args.shard_tokens)
    target_total = args.max_tokens + args.val_tokens
    documents = 0
    skipped = 0

    print("Preparing Hugging Face LM dataset")
    print(f"  dataset_name:   {args.dataset_name}")
    print(f"  dataset_config: {args.dataset_config}")
    print(f"  split:          {args.split}")
    print(f"  format:         {args.format}")
    print(f"  text_field:     {text_field or '(format-derived)'}")
    print(f"  target tokens:  train={args.max_tokens:,}, val={args.val_tokens:,}")
    print(f"  dtype:          {dtype_name}")

    with tqdm(total=target_total, unit="tok", desc="Tokenizing", dynamic_ncols=True) as pbar:
        for example in iter_examples(iterator, first_example):
            if val_writer.total_tokens >= args.val_tokens and train_writer.total_tokens >= args.max_tokens:
                break
            text = extract_text(dict(example), args, text_field)
            if not text:
                skipped += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids:
                skipped += 1
                continue
            ids.append(eos_id)
            documents += 1

            cursor = 0
            written = 0
            if val_writer.total_tokens < args.val_tokens:
                remaining_val = args.val_tokens - val_writer.total_tokens
                wrote = val_writer.write(ids[cursor:], limit=remaining_val)
                cursor += wrote
                written += wrote
            if cursor < len(ids) and train_writer.total_tokens < args.max_tokens:
                remaining_train = args.max_tokens - train_writer.total_tokens
                written += train_writer.write(ids[cursor:], limit=remaining_train)
            if written:
                pbar.update(written)

    train_writer.flush()
    val_writer.flush()
    if val_writer.total_tokens == 0 or train_writer.total_tokens == 0:
        raise ValueError("No train or validation tokens were written.")

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "format": args.format,
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": int(vocab_size),
        "dtype": dtype_name,
        "train_tokens": int(train_writer.total_tokens),
        "val_tokens": int(val_writer.total_tokens),
        "shard_tokens": int(args.shard_tokens),
        "documents": documents,
        "skipped_examples": skipped,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
