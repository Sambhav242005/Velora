from __future__ import annotations

import argparse
import sys

import torch
from tokenizers import Tokenizer

from src.model import GPT, ModelConfig


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    if cfg.get("vocab_size") == "auto":
        cfg["vocab_size"] = ckpt["config"].get("tokenizer_vocab_size", 4096)
    # Better source: infer vocab size from embedding shape.
    cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
    model = GPT(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
