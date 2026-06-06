from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tokenizers import Tokenizer
from tqdm import tqdm

from prepare_sft import (
    IGNORE_INDEX,
    encode_example,
    find_eos_id,
    load_dataset_checked,
    parse_bool,
    positive_int,
    save_split,
)


def take_examples(dataset: Iterable[Dict[str, Any]], limit: int) -> list[Dict[str, Any]]:
    examples = []
    for example in dataset:
        examples.append(dict(example))
        if len(examples) >= limit:
            break
    return examples


def load_source(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    streaming: bool,
    limit: int,
    source: str,
) -> list[Dict[str, Any]]:
    if limit <= 0:
        return []
    dataset = load_dataset_checked(dataset_name, dataset_config, split, streaming)
    examples = take_examples(dataset, limit)
    for example in examples:
        example["_source"] = source
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small reasoning-focused SFT mix.")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--out_dir", default="data/sft/reasoning_polish_1024")
    parser.add_argument("--max_seq_len", type=positive_int, default=1024)
    parser.add_argument("--train_examples", type=positive_int, default=50_000)
    parser.add_argument("--val_examples", type=positive_int, default=2_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--streaming", type=parse_bool, default=True)
    parser.add_argument("--smoltalk_dataset_name", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--smoltalk_dataset_config", default=None)
    parser.add_argument("--smoltalk_split", default="train")
    parser.add_argument("--smoltalk_examples", type=positive_int, default=60_000)
    parser.add_argument("--gsm8k_dataset_name", default="openai/gsm8k")
    parser.add_argument("--gsm8k_dataset_config", default="main")
    parser.add_argument("--gsm8k_split", default="train")
    parser.add_argument("--gsm8k_examples", type=positive_int, default=8_000)
    parser.add_argument("--reasoning_dataset_name", default=None)
    parser.add_argument("--reasoning_dataset_config", default=None)
    parser.add_argument("--reasoning_split", default="train")
    parser.add_argument("--reasoning_examples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output folder already exists: {out_dir}. Pass --overwrite to replace it.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_examples = args.train_examples + args.val_examples
    examples: list[Dict[str, Any]] = []
    examples.extend(
        load_source(
            args.smoltalk_dataset_name,
            args.smoltalk_dataset_config,
            args.smoltalk_split,
            args.streaming,
            args.smoltalk_examples,
            "smoltalk",
        )
    )
    examples.extend(
        load_source(
            args.gsm8k_dataset_name,
            args.gsm8k_dataset_config,
            args.gsm8k_split,
            args.streaming,
            args.gsm8k_examples,
            "gsm8k",
        )
    )
    if args.reasoning_dataset_name and args.reasoning_examples > 0:
        examples.extend(
            load_source(
                args.reasoning_dataset_name,
                args.reasoning_dataset_config,
                args.reasoning_split,
                args.streaming,
                args.reasoning_examples,
                "reasoning",
            )
        )

    if len(examples) < 2:
        raise ValueError("No source examples were loaded.")
    random.Random(args.seed).shuffle(examples)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = find_eos_id(tokenizer)
    encoded = []
    skipped = 0
    for example in tqdm(examples, desc="Encoding", dynamic_ncols=True):
        item = encode_example(tokenizer, eos_id, example, args.max_seq_len)
        if item is None:
            skipped += 1
            continue
        encoded.append(item)
        if len(encoded) >= target_examples:
            break
    if len(encoded) < target_examples:
        raise ValueError(f"Only encoded {len(encoded):,} usable examples; need {target_examples:,}.")

    val_items = encoded[:args.val_examples]
    train_items = encoded[args.val_examples:target_examples]
    save_split(out_dir, "train", [x for x, _ in train_items], [y for _, y in train_items])
    save_split(out_dir, "val", [x for x, _ in val_items], [y for _, y in val_items])

    meta = {
        "format": "reasoning_sft_mix",
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "max_seq_len": int(args.max_seq_len),
        "ignore_index": IGNORE_INDEX,
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "skipped_examples": skipped,
        "sources": {
            "smoltalk": {
                "dataset_name": args.smoltalk_dataset_name,
                "dataset_config": args.smoltalk_dataset_config,
                "split": args.smoltalk_split,
                "requested_examples": args.smoltalk_examples,
            },
            "gsm8k": {
                "dataset_name": args.gsm8k_dataset_name,
                "dataset_config": args.gsm8k_dataset_config,
                "split": args.gsm8k_split,
                "requested_examples": args.gsm8k_examples,
            },
            "reasoning": {
                "dataset_name": args.reasoning_dataset_name,
                "dataset_config": args.reasoning_dataset_config,
                "split": args.reasoning_split,
                "requested_examples": args.reasoning_examples,
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
