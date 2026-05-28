from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def iter_documents(input_dir: Path) -> Iterable[str]:
    for path in sorted(input_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Split on blank lines to get rough documents.
        for doc in text.split("\n\n"):
            doc = doc.strip()
            if doc:
                yield doc


def stable_val_split(doc: str, val_fraction: float) -> bool:
    h = hashlib.md5(doc.encode("utf-8", errors="ignore")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def flush_tokens(buffer: List[int], file_obj, dtype) -> int:
    if not buffer:
        return 0
    arr = np.asarray(buffer, dtype=dtype)
    arr.tofile(file_obj)
    n = len(buffer)
    buffer.clear()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--flush_tokens", type=int, default=1_000_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    dtype_name = "uint16" if dtype == np.uint16 else "uint32"

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_tokens = 0
    val_tokens = 0
    train_buffer: List[int] = []
    val_buffer: List[int] = []

    doc_count = 0
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for doc in tqdm(iter_documents(input_dir), desc="Tokenizing"):
            doc_count += 1
            ids = tok.encode(doc).ids
            if stable_val_split(doc, args.val_fraction):
                val_buffer.extend(ids)
                if len(val_buffer) >= args.flush_tokens:
                    val_tokens += flush_tokens(val_buffer, val_f, dtype)
            else:
                train_buffer.extend(ids)
                if len(train_buffer) >= args.flush_tokens:
                    train_tokens += flush_tokens(train_buffer, train_f, dtype)
        train_tokens += flush_tokens(train_buffer, train_f, dtype)
        val_tokens += flush_tokens(val_buffer, val_f, dtype)

    if doc_count == 0:
        raise FileNotFoundError(f"No text documents found in {input_dir}")

    # Ensure validation is not empty for tiny sample data.
    if val_tokens < 1000:
        print("Validation split is very small. Creating fallback val.bin from start of train.bin.")
        arr = np.memmap(train_path, dtype=dtype, mode="r")
        val_copy = np.asarray(arr[:min(10000, len(arr))], dtype=dtype)
        val_copy.tofile(val_path)
        val_tokens = len(val_copy)

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "tokenizer": str(Path(args.tokenizer).resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
