from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

import torch
from tokenizers import Tokenizer

from src.guided import add_regex_guidance_args, build_regex_logits_processor
from src.json_guided import add_json_guidance_args, build_json_logits_processor
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


def first_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for pos, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def strict_json(text: str) -> str:
    text = text.strip().replace("<unk>", "{")
    if not text.startswith("{") and text.startswith('"'):
        text = "{" + text
    candidate = first_balanced_json(text)
    if candidate is None:
        return text
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return candidate
    return candidate


def strict_xml(text: str) -> str:
    text = text.strip()
    match = re.search(r"<([A-Za-z_][\w:.-]*)(?:\s[^>]*)?>.*?</\1>", text, flags=re.DOTALL)
    if match is None:
        return text
    candidate = match.group(0)
    try:
        ET.fromstring(candidate)
    except ET.ParseError:
        return candidate
    return candidate


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate strict JSON/XML from an instruction checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--format", choices=["json", "xml"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    add_regex_guidance_args(parser)
    add_json_guidance_args(parser)
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
    if args.regex:
        logits_processor, eos_token_id = build_regex_logits_processor(args, tok, [x.size(1)])
    elif args.format == "json" and not args.no_json_guide:
        logits_processor, eos_token_id = build_json_logits_processor(args, tok, [x.size(1)])
    else:
        logits_processor, eos_token_id = None, None
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

    text = tok.decode(y[0].tolist(), skip_special_tokens=False)
    response = extract_response(text)
    if args.format == "json":
        print(strict_json(response))
    else:
        print(strict_xml(response))


if __name__ == "__main__":
    main()
