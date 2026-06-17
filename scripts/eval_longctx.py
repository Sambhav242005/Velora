from __future__ import annotations
import argparse, random, re, sys
import numpy as np
import torch
from tokenizers import Tokenizer
sys.path.insert(0, ".")
from src.model import GPT, ModelConfig
from generate_instruct import format_prompt, extract_response


def normalize_state_dict(state_dict):
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in state_dict):
        return {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"]["model"])
    cfg["vocab_size"] = ckpt["model"]["tok_embeddings.weight"].shape[0]
    model = GPT(ModelConfig(**cfg)).to(device)
    model.load_state_dict(normalize_state_dict(ckpt["model"])); model.eval()
    return model


def gsm8k_em(model, tok, device, n=200):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    correct = 0
    for ex in ds:
        prompt = format_prompt(ex["question"], "")
        ids = tok.encode(prompt, add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=256, temperature=0.0, top_k=0, top_p=None)
        out = extract_response(tok.decode(y[0].tolist()))
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        pred = re.findall(r"-?\d[\d,]*", out)
        pred = pred[-1].replace(",", "") if pred else ""
        correct += int(pred == gold)
    return correct / max(1, len(ds))


def ppl_by_position(model, tok, device, val_npy, block, bins=8):
    data = np.load(val_npy, mmap_mode="r")
    n = (len(data) // (block + 1)) * (block + 1)
    chunks = np.asarray(data[:n], dtype=np.int64).reshape(-1, block + 1)[:64]
    x = torch.from_numpy(chunks[:, :-1]).to(device)
    y = torch.from_numpy(chunks[:, 1:]).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).reshape(x.size(0), -1).mean(0)
    edges = torch.linspace(0, loss.numel(), bins + 1).long()
    return [loss[edges[i]:edges[i + 1]].mean().item() for i in range(bins)]


def passkey_recall(model, tok, device, n=20, context_tokens=None):
    rng = random.Random(1337)
    max_prompt_tokens = min(context_tokens or model.config.block_size, model.config.block_size)
    max_prompt_tokens = max(1, max_prompt_tokens - 16)
    filler_ids = tok.encode(
        " Neutral background sentence with no digits and no useful clue.",
        add_special_tokens=False,
    ).ids
    correct = 0
    for _ in range(n):
        key = str(rng.randint(10000, 99999))
        prefix_ids = tok.encode(f"The pass key is {key}.\n", add_special_tokens=False).ids
        question_ids = tok.encode(
            "\nWhat is the pass key? Answer with only the number.\n",
            add_special_tokens=False,
        ).ids
        ids = list(prefix_ids)
        while len(ids) + len(filler_ids) + len(question_ids) <= max_prompt_tokens:
            ids.extend(filler_ids)
        ids.extend(question_ids)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=16, temperature=0.0, top_k=0, top_p=None)
        generated = tok.decode(y[0, len(ids):].tolist())
        pred = re.findall(r"\d+", generated)
        correct += int(bool(pred) and pred[0] == key)
    return correct / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--gsm8k", type=int, default=200)
    p.add_argument("--val_npy", default=None, help="a val shard .npy to probe ppl-by-position")
    p.add_argument("--block", type=int, default=16384)
    p.add_argument("--passkey", type=int, default=0, help="number of synthetic passkey recall trials")
    p.add_argument("--passkey_context", type=int, default=None, help="prompt length cap for passkey recall")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tok = Tokenizer.from_file(args.tokenizer)
    if args.gsm8k > 0:
        print(f"GSM8K exact-match ({args.gsm8k}): {gsm8k_em(model, tok, device, args.gsm8k):.3f}")
    if args.val_npy:
        bins = ppl_by_position(model, tok, device, args.val_npy, args.block)
        print("mean loss by position bin (early -> late):")
        print("  " + "  ".join(f"{b:.3f}" for b in bins))
        print("  (late bins should be <= early bins if long context is being used)")
    if args.passkey > 0:
        print(
            f"Passkey recall ({args.passkey}): "
            f"{passkey_recall(model, tok, device, args.passkey, args.passkey_context):.3f}"
        )


if __name__ == "__main__":
    main()
