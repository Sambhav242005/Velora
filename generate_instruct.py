from __future__ import annotations

import argparse
import re
import sys

import torch
from tokenizers import Tokenizer

from src.guided import add_regex_guidance_args, build_regex_logits_processor
from src.inference import checkpoint_error_message, checkpoint_exists
from src.model import GPT, ModelConfig


def format_prompt(instruction: str, context: str = "") -> str:
    instruction = instruction.strip()
    context = context.strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extract_response(text: str) -> str:
    marker = "### Response:"
    if marker in text:
        text = text.split(marker, 1)[1]
    for stop_marker in ("\n### Instruction:", "\n### Input:", "\n### Response:"):
        if stop_marker in text:
            text = text.split(stop_marker, 1)[0]
    return text.strip()


def trim_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text.strip()
    sentence_ends = list(re.finditer(r"[.!?](?=(?:\s+[A-Z0-9#\"']|$|[A-Z][a-z]))", text))
    if len(sentence_ends) < max_sentences:
        return text.strip()
    return text[: sentence_ends[max_sentences - 1].end()].strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_answer_sentences", type=int, default=4)
    parser.add_argument("--full_output", action="store_true")
    add_regex_guidance_args(parser)
    args = parser.parse_args()

    if not checkpoint_exists(args.checkpoint):
        parser.error(checkpoint_error_message(args.checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"]["model"])
    cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
    model = GPT(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    prompt = format_prompt(args.instruction, args.input)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits_processor, eos_token_id = build_regex_logits_processor(args, tok, [x.size(1)])
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            logits_processor=logits_processor,
            eos_token_id=eos_token_id,
        )
    text = tok.decode(y[0].tolist())
    if not args.full_output:
        text = trim_sentences(extract_response(text), args.max_answer_sentences)
    print(text)


if __name__ == "__main__":
    main()
