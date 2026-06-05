from __future__ import annotations

import argparse
import sys

import torch
from tokenizers import Tokenizer

from src.model import GPT, ModelConfig
from generate_instruct import format_prompt, extract_response, trim_sentences


def load_model(path: str, device: torch.device) -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"]["model"])
    cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
    model = GPT(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def build_history(turns: list[tuple[str, str]]) -> str:
    # Matches the history format produced by scripts/prepare_sft.py: lines of
    # "User: ..." / "Assistant: ..." that get placed in the ### Input: field.
    return "\n".join(f"{role}: {content}" for role, content in turns)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Interactive multi-turn chat REPL for an SFT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_answer_sentences", type=int, default=0, help="0 = no sentence trimming")
    parser.add_argument("--max_history_turns", type=int, default=6, help="user+assistant exchanges kept as context")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model = load_model(args.checkpoint, device)
    tok = Tokenizer.from_file(args.tokenizer)
    block = model.config.block_size

    print("Chat ready. Commands: /reset clears history, /exit quits.\n")
    turns: list[tuple[str, str]] = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            turns = []
            print("(history cleared)\n")
            continue

        history = build_history(turns[-2 * args.max_history_turns:])
        prompt = format_prompt(user, history)
        ids = tok.encode(prompt, add_special_tokens=False).ids
        ids = ids[-(block - args.max_new_tokens):]  # keep room for the reply
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
        reply = extract_response(tok.decode(y[0].tolist()))
        if args.max_answer_sentences > 0:
            reply = trim_sentences(reply, args.max_answer_sentences)
        print(f"Bot: {reply}\n")
        turns.append(("User", user))
        turns.append(("Assistant", reply))


if __name__ == "__main__":
    main()
