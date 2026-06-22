from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from tokenizers import Tokenizer
from tqdm import tqdm

from prepare_sft import IGNORE_INDEX, encode_example, find_eos_id, positive_int, save_split


NAMES = [
    "Asha Rao",
    "Mira Shah",
    "Neel Verma",
    "Sara Khan",
    "Ravi Iyer",
    "Anika Das",
    "Dev Patel",
    "Leah Stone",
]
PRODUCTS = ["notebook", "sensor", "adapter", "keyboard", "monitor", "router", "cable", "battery"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Jaipur", "Kolkata"]
STATUSES = ["queued", "running", "complete", "failed", "cancelled"]


def parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("At least one format is required.")
    invalid = sorted(set(formats) - {"json", "xml"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported formats: {', '.join(invalid)}")
    return formats


def pick(rng: random.Random, values: list[str]) -> str:
    return rng.choice(values)


def json_person(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, NAMES)
    payload = {
        "type": "person",
        "name": name,
        "age": rng.randint(18, 72),
        "city": pick(rng, CITIES),
        "active": rng.choice([True, False]),
        "skills": rng.sample(["python", "sql", "design", "writing", "testing", "ml"], k=3),
    }
    instruction = f"Return only JSON for a user profile named {name}."
    return instruction, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def json_order(rng: random.Random) -> tuple[str, str]:
    count = rng.randint(1, 4)
    items = []
    total = 0.0
    for _ in range(count):
        qty = rng.randint(1, 5)
        price = round(rng.uniform(3.5, 240.0), 2)
        total += qty * price
        items.append({"sku": f"SKU-{rng.randint(1000, 9999)}", "name": pick(rng, PRODUCTS), "qty": qty, "price": price})
    order_id = f"ORD-{rng.randint(10000, 99999)}"
    payload = {"order_id": order_id, "currency": "USD", "items": items, "total": round(total, 2)}
    instruction = f"Create only a JSON order object for order {order_id}."
    return instruction, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def json_status(rng: random.Random) -> tuple[str, str]:
    job_id = f"job_{rng.randint(100000, 999999)}"
    payload = {
        "job_id": job_id,
        "status": pick(rng, STATUSES),
        "progress": rng.randint(0, 100),
        "retryable": rng.choice([True, False]),
        "errors": [] if rng.random() < 0.7 else [{"code": "E_TIMEOUT", "message": "operation timed out"}],
    }
    instruction = f"Respond only with JSON showing the status for {job_id}."
    return instruction, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def json_config(rng: random.Random) -> tuple[str, str]:
    service = pick(rng, ["api", "worker", "search", "billing", "gateway"])
    payload = {
        "service": service,
        "replicas": rng.randint(1, 8),
        "debug": rng.choice([True, False]),
        "limits": {"cpu": f"{rng.randint(1, 8)}", "memory_gb": rng.choice([2, 4, 8, 16])},
        "features": {"cache": rng.choice([True, False]), "metrics": True},
    }
    instruction = f"Output only JSON config for the {service} service."
    return instruction, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def xml_person(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, NAMES)
    city = pick(rng, CITIES)
    active = str(rng.choice([True, False])).lower()
    xml = (
        f'<person active="{active}">'
        f"<name>{escape(name)}</name>"
        f"<age>{rng.randint(18, 72)}</age>"
        f"<city>{escape(city)}</city>"
        "</person>"
    )
    instruction = f"Return only XML for a user profile named {name}."
    return instruction, xml


def xml_order(rng: random.Random) -> tuple[str, str]:
    order_id = f"ORD-{rng.randint(10000, 99999)}"
    parts = [f'<order id="{order_id}" currency="USD">']
    total = 0.0
    for _ in range(rng.randint(1, 4)):
        qty = rng.randint(1, 5)
        price = round(rng.uniform(3.5, 240.0), 2)
        total += qty * price
        parts.append(
            f'<item sku="SKU-{rng.randint(1000, 9999)}" qty="{qty}" price="{price}">'
            f"{escape(pick(rng, PRODUCTS))}</item>"
        )
    parts.append(f"<total>{round(total, 2)}</total></order>")
    instruction = f"Create only an XML order for order {order_id}."
    return instruction, "".join(parts)


def xml_status(rng: random.Random) -> tuple[str, str]:
    job_id = f"job_{rng.randint(100000, 999999)}"
    status = pick(rng, STATUSES)
    xml = (
        f'<job id="{job_id}">'
        f"<status>{status}</status>"
        f"<progress>{rng.randint(0, 100)}</progress>"
        f"<retryable>{str(rng.choice([True, False])).lower()}</retryable>"
        "</job>"
    )
    instruction = f"Respond only with XML showing the status for {job_id}."
    return instruction, xml


def xml_config(rng: random.Random) -> tuple[str, str]:
    service = pick(rng, ["api", "worker", "search", "billing", "gateway"])
    xml = (
        f'<config service="{service}">'
        f"<replicas>{rng.randint(1, 8)}</replicas>"
        f"<debug>{str(rng.choice([True, False])).lower()}</debug>"
        f'<limits cpu="{rng.randint(1, 8)}" memory_gb="{rng.choice([2, 4, 8, 16])}" />'
        "</config>"
    )
    instruction = f"Output only XML config for the {service} service."
    return instruction, xml


JSON_BUILDERS = [json_person, json_order, json_status, json_config]
XML_BUILDERS = [xml_person, xml_order, xml_status, xml_config]


def make_example(rng: random.Random, formats: list[str]) -> dict[str, str]:
    fmt = pick(rng, formats)
    if fmt == "json":
        instruction, response = pick(rng, JSON_BUILDERS)(rng)
    else:
        instruction, response = pick(rng, XML_BUILDERS)(rng)
    return {"instruction": instruction, "input": "", "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare synthetic JSON/XML-only SFT arrays.")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--out_dir", default="data/sft/structured_outputs_1024")
    parser.add_argument("--max_seq_len", type=positive_int, default=1024)
    parser.add_argument("--train_examples", type=positive_int, default=80_000)
    parser.add_argument("--val_examples", type=positive_int, default=2_000)
    parser.add_argument("--formats", type=parse_formats, default=["json", "xml"], help="json, xml, or json,xml")
    parser.add_argument("--seed", type=int, default=1337)
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

    rng = random.Random(args.seed)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = find_eos_id(tokenizer)
    total_examples = args.train_examples + args.val_examples

    encoded = []
    skipped = 0
    for _ in tqdm(range(total_examples), desc="Encoding", dynamic_ncols=True):
        item = encode_example(tokenizer, eos_id, make_example(rng, args.formats), args.max_seq_len)
        if item is None:
            skipped += 1
            continue
        encoded.append(item)
    if len(encoded) < total_examples:
        raise ValueError(f"Only encoded {len(encoded):,} usable examples; need {total_examples:,}.")

    val_items = encoded[: args.val_examples]
    train_items = encoded[args.val_examples :]
    save_split(out_dir, "train", [x for x, _ in train_items], [y for _, y in train_items])
    save_split(out_dir, "val", [x for x, _ in val_items], [y for _, y in val_items])

    meta: dict[str, Any] = {
        "format": "synthetic_structured_outputs",
        "formats": args.formats,
        "tokenizer": str(tokenizer_path.resolve()),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "max_seq_len": int(args.max_seq_len),
        "ignore_index": IGNORE_INDEX,
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "skipped_examples": skipped,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
