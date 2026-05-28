from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


def iter_text_files(input_dir: Path):
    for path in sorted(input_dir.rglob("*.txt")):
        yield str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_text_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    tokenizer.save(str(out_dir / "tokenizer.json"))
    print(f"Saved tokenizer to {out_dir / 'tokenizer.json'}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
