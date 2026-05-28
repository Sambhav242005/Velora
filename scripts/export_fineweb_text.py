from __future__ import annotations

import argparse
import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MANY_BLANK_LINES = re.compile(r"\n{3,}")


def positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative")
    return parsed


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub(" ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = MANY_BLANK_LINES.sub("\n\n", text)
    return text


def load_streaming_dataset(dataset_name: str, dataset_config: str, split: str):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets with `pip install -r requirements.txt`.") from exc

    try:
        return load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    except Exception as exc:
        try:
            configs = get_dataset_config_names(dataset_name)
            if dataset_config not in configs:
                print(f"Dataset config {dataset_config!r} was not found for {dataset_name}.")
                print("Available configs:")
                for config in configs:
                    print(f"  - {config}")
        except Exception as config_exc:
            print(f"Could not fetch available configs for {dataset_name}: {config_exc}")
        raise RuntimeError(
            f"Could not load dataset={dataset_name!r}, config={dataset_config!r}, split={split!r} with streaming=True."
        ) from exc


def get_text(example: Dict[str, Any], text_field: str) -> Optional[str]:
    if text_field not in example:
        raise ValueError(f"Text field {text_field!r} not found. Available fields: {sorted(example.keys())}")
    value = example[text_field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Text field {text_field!r} exists but is not a string.")
    return value


def close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def quiet_hf_cleanup_logs() -> None:
    for name in ("huggingface_hub", "datasets"):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a bounded FineWeb/FineWeb-Edu text corpus for tokenizer training.")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_file", default="data/tokenizer_corpus/fineweb_sample.txt")
    parser.add_argument("--max_chars", type=positive_int, default=200_000_000)
    parser.add_argument("--max_docs", type=non_negative_int, default=0, help="0 means no document-count limit.")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_file = Path(args.out_file)
    if out_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {out_file}. Pass --overwrite to replace it.")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("Exporting streaming text corpus")
    print(f"  dataset_name:   {args.dataset_name}")
    print(f"  dataset_config: {args.dataset_config}")
    print(f"  split:          {args.split}")
    print(f"  text_field:     {args.text_field}")
    print(f"  out_file:       {out_file}")
    print(f"  max_chars:      {args.max_chars:,}")
    print(f"  max_docs:       {args.max_docs:,}" if args.max_docs else "  max_docs:       unlimited")

    dataset = load_streaming_dataset(args.dataset_name, args.dataset_config, args.split)
    iterator = iter(dataset)

    docs_written = 0
    chars_written = 0
    try:
        with out_file.open("w", encoding="utf-8", newline="\n") as f:
            for example in iterator:
                if chars_written >= args.max_chars:
                    break
                if args.max_docs and docs_written >= args.max_docs:
                    break

                text = get_text(example, args.text_field)
                if not text:
                    continue
                text = clean_text(text)
                if not text:
                    continue

                remaining = args.max_chars - chars_written
                separator = "\n\n" if docs_written > 0 else ""
                required = len(separator) + len(text)
                if required > remaining:
                    if remaining <= len(separator):
                        break
                    fragment = text[:remaining - len(separator)].rstrip()
                    if not fragment:
                        break
                    f.write(separator)
                    f.write(fragment)
                    chars_written += len(separator) + len(fragment)
                    docs_written += 1
                    break

                f.write(separator)
                f.write(text)
                chars_written += required
                docs_written += 1

                if docs_written % 1000 == 0:
                    print(f"Progress: docs={docs_written:,}, chars={chars_written:,}/{args.max_chars:,}")
    finally:
        quiet_hf_cleanup_logs()
        close_iterator(iterator)

    print(f"Done: wrote docs={docs_written:,}, chars={chars_written:,} to {out_file}")
    if docs_written == 0:
        raise ValueError("No documents were written. Check dataset/config/split/text_field.")


if __name__ == "__main__":
    main()
