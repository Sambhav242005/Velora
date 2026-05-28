from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


IGNORE_INDEX = -100


def positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def load_dataset_checked(dataset_name: str, dataset_config: Optional[str], split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets with `pip install -r requirements.txt`.") from exc

    kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    return load_dataset(dataset_name, **kwargs)


def find_eos_id(tokenizer: Tokenizer) -> int:
    for token in ("<eos>", "</s>", "<|endoftext|>", "[EOS]", "[SEP]"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return int(token_id)
    raise ValueError("No EOS token found in tokenizer.")


def field_text(example: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = example.get(name)
        if isinstance(value, str):
            return value.strip()
    return ""


def message_role(message: Dict[str, Any]) -> str:
    role = str(message.get("role", message.get("from", ""))).strip().lower()
    if role in {"user", "human"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def message_content(message: Dict[str, Any]) -> str:
    return str(message.get("content", message.get("value", ""))).strip()


def messages_to_instruction_response(messages: list[Any]) -> tuple[str, str, str]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message_role(message)
        content = message_content(message)
        if role and content:
            normalized.append((role, content))

    assistant_indices = [idx for idx, (role, _) in enumerate(normalized) if role == "assistant"]
    if not assistant_indices:
        return "", "", ""

    assistant_idx = assistant_indices[-1]
    response = normalized[assistant_idx][1]
    user_indices = [
        idx
        for idx, (role, _) in enumerate(normalized[:assistant_idx])
        if role == "user"
    ]
    if not user_indices:
        return "", "", ""

    user_idx = user_indices[-1]
    instruction = normalized[user_idx][1]
    history = []
    for role, content in normalized[:user_idx]:
        if role == "system":
            history.append(f"System: {content}")
        elif role == "user":
            history.append(f"User: {content}")
        elif role == "assistant":
            history.append(f"Assistant: {content}")
    return instruction, "\n".join(history), response


def example_to_instruction_response(example: Dict[str, Any]) -> tuple[str, str, str]:
    messages = example.get("messages")
    if messages is None:
        messages = example.get("conversations")
    if isinstance(messages, list):
        return messages_to_instruction_response(messages)

    instruction = field_text(example, "instruction", "prompt", "question")
    context = field_text(example, "input", "context")
    response = field_text(example, "output", "response", "answer", "completion")
    return instruction, context, response


def format_prompt(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def encode_example(
    tokenizer: Tokenizer,
    eos_id: int,
    example: Dict[str, Any],
    max_seq_len: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    instruction, context, response = example_to_instruction_response(example)
    if not instruction or not response:
        return None

    prompt_ids = tokenizer.encode(format_prompt(instruction, context), add_special_tokens=False).ids
    response_ids = tokenizer.encode(response.strip(), add_special_tokens=False).ids
    if not prompt_ids or not response_ids:
        return None

    max_full_len = max_seq_len + 1
    room_for_response = max_full_len - len(prompt_ids) - 1
    if room_for_response <= 0:
        prompt_ids = prompt_ids[: max_seq_len // 2]
        room_for_response = max_full_len - len(prompt_ids) - 1
    if room_for_response <= 0:
        return None

    response_ids = response_ids[:room_for_response]
    full_ids = prompt_ids + response_ids + [eos_id]
    if len(full_ids) < 2:
        return None

    x_ids = full_ids[:-1][:max_seq_len]
    y_ids = full_ids[1:][:max_seq_len]
    prompt_len = min(len(prompt_ids), len(x_ids))
    first_loss_idx = max(0, prompt_len - 1)

    input_ids = np.zeros((max_seq_len,), dtype=np.uint16)
    labels = np.full((max_seq_len,), IGNORE_INDEX, dtype=np.int32)
    input_ids[: len(x_ids)] = np.asarray(x_ids, dtype=np.uint16)
    labels[first_loss_idx : len(y_ids)] = np.asarray(y_ids[first_loss_idx:], dtype=np.int32)
    return input_ids, labels


def materialize_examples(dataset: Iterable[Dict[str, Any]], max_examples: Optional[int]) -> list[Dict[str, Any]]:
    examples = []
    for example in dataset:
        examples.append(dict(example))
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def save_split(out_dir: Path, split: str, inputs: list[np.ndarray], labels: list[np.ndarray]) -> None:
    if not inputs:
        raise ValueError(f"No {split} examples were encoded.")
    np.save(out_dir / f"{split}_input_ids.npy", np.stack(inputs))
    np.save(out_dir / f"{split}_labels.npy", np.stack(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare instruction SFT arrays.")
    parser.add_argument("--dataset_name", default="yahma/alpaca-cleaned")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--out_dir", default="data/sft/alpaca_cleaned_512")
    parser.add_argument("--max_seq_len", type=positive_int, default=512)
    parser.add_argument("--val_fraction", type=float, default=0.03)
    parser.add_argument("--max_examples", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--streaming", type=parse_bool, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("--val_fraction must be > 0 and < 0.5")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output folder already exists: {out_dir}. Pass --overwrite to replace it.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing SFT dataset")
    print(f"  dataset_name:   {args.dataset_name}")
    print(f"  dataset_config: {args.dataset_config}")
    print(f"  split:          {args.split}")
    print(f"  tokenizer:      {tokenizer_path}")
    print(f"  out_dir:        {out_dir}")
    print(f"  max_seq_len:    {args.max_seq_len}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = find_eos_id(tokenizer)
    dataset = load_dataset_checked(args.dataset_name, args.dataset_config, args.split, args.streaming)
    examples = materialize_examples(dataset, args.max_examples)
    random.Random(args.seed).shuffle(examples)

    encoded = []
    skipped = 0
    for example in tqdm(examples, desc="Encoding", dynamic_ncols=True):
        item = encode_example(tokenizer, eos_id, example, args.max_seq_len)
        if item is None:
            skipped += 1
            continue
        encoded.append(item)
    if len(encoded) < 2:
        raise ValueError("Too few usable SFT examples.")

    val_count = max(1, int(len(encoded) * args.val_fraction))
    val_items = encoded[:val_count]
    train_items = encoded[val_count:]

    save_split(out_dir, "train", [x for x, _ in train_items], [y for _, y in train_items])
    save_split(out_dir, "val", [x for x, _ in val_items], [y for _, y in val_items])

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "max_seq_len": int(args.max_seq_len),
        "ignore_index": IGNORE_INDEX,
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "skipped_examples": skipped,
        "format": "alpaca_instruction_input_response",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
