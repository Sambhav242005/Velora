from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from tokenizers import Tokenizer
from tqdm import tqdm

from prepare_sft import IGNORE_INDEX, encode_example, find_eos_id, positive_int, save_split


NAMES = [
    "Ravi Iyer",
    "Asha Rao",
    "Mira Shah",
    "Neel Verma",
    "Sara Khan",
    "Anika Das",
    "Dev Patel",
    "Leah Stone",
]
SKILLS = ["Python", "ML", "SQL", "testing", "design", "writing", "data analysis", "API development"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Jaipur", "Kolkata"]
PRODUCTS = ["Laptop", "Notebook", "Sensor", "Adapter", "Keyboard", "Monitor", "Router", "Battery"]
STATUSES = ["queued", "running", "complete", "failed", "cancelled"]

JSON_PROMPT_SUFFIXES = [
    "Return exactly one valid JSON object. Start with { and end with }. No markdown. No prose.",
    "Output only JSON. The first character must be { and the last character must be }.",
    "Respond with a single compact JSON object and nothing else.",
    "Give only parseable JSON. Do not explain.",
]
XML_PROMPT_SUFFIXES = [
    "Return exactly one valid XML document. Start with the root tag and end with the matching closing tag. No markdown. No prose.",
    "Output only XML. Do not include explanations.",
    "Respond with a single compact XML document and nothing else.",
    "Give only parseable XML. Do not explain.",
]


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


def compact_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    parsed = json.loads(text)
    if parsed != payload:
        raise ValueError("JSON round trip changed payload.")
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("JSON target does not have strict object boundaries.")
    return text


def compact_xml(text: str) -> str:
    ET.fromstring(text)
    if not (text.startswith("<") and text.endswith(">")):
        raise ValueError("XML target does not have strict tag boundaries.")
    return text


def json_user_custom(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, NAMES)
    age = rng.randint(18, 72)
    skills = rng.sample(SKILLS, k=rng.randint(2, 4))
    payload = {"name": name, "age": age, "skills": skills}
    instruction = (
        f"Return a JSON object for a user named {name}, age {age}, "
        f"with skills {', '.join(skills[:-1])} and {skills[-1]}. "
        f"{pick(rng, JSON_PROMPT_SUFFIXES)}"
    )
    return instruction, compact_json(payload)


def json_user_schema(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, NAMES)
    payload = {
        "type": "person",
        "name": name,
        "age": rng.randint(18, 72),
        "city": pick(rng, CITIES),
        "active": rng.choice([True, False]),
        "skills": rng.sample(SKILLS, k=3),
    }
    instruction = (
        f"Return only JSON for a user profile named {name}. "
        f"Use keys type, name, age, city, active, skills. {pick(rng, JSON_PROMPT_SUFFIXES)}"
    )
    return instruction, compact_json(payload)


def json_product(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, PRODUCTS)
    price = rng.randint(25, 1800)
    in_stock = rng.choice([True, False])
    payload = {"product": name, "price": price, "in_stock": in_stock}
    instruction = f"Return a JSON object for product {name} with price {price} and in_stock {str(in_stock).lower()}. {pick(rng, JSON_PROMPT_SUFFIXES)}"
    return instruction, compact_json(payload)


def json_status(rng: random.Random) -> tuple[str, str]:
    job_id = f"job_{rng.randint(100000, 999999)}"
    payload = {
        "job_id": job_id,
        "status": pick(rng, STATUSES),
        "progress": rng.randint(0, 100),
        "retryable": rng.choice([True, False]),
        "errors": [] if rng.random() < 0.75 else [{"code": "E_TIMEOUT", "message": "operation timed out"}],
    }
    instruction = f"Return status for {job_id} as JSON with keys job_id, status, progress, retryable, errors. {pick(rng, JSON_PROMPT_SUFFIXES)}"
    return instruction, compact_json(payload)


def json_order(rng: random.Random) -> tuple[str, str]:
    order_id = f"ORD-{rng.randint(10000, 99999)}"
    items = []
    total = 0.0
    for _ in range(rng.randint(1, 4)):
        qty = rng.randint(1, 5)
        price = round(rng.uniform(3.5, 240.0), 2)
        total += qty * price
        items.append({"sku": f"SKU-{rng.randint(1000, 9999)}", "name": pick(rng, PRODUCTS), "qty": qty, "price": price})
    payload = {"order_id": order_id, "currency": "USD", "items": items, "total": round(total, 2)}
    instruction = f"Create a JSON order object for order {order_id}. {pick(rng, JSON_PROMPT_SUFFIXES)}"
    return instruction, compact_json(payload)


def xml_user_custom(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, NAMES)
    age = rng.randint(18, 72)
    skills = rng.sample(SKILLS, k=rng.randint(2, 4))
    skill_nodes = "".join(f"<skill>{escape(skill)}</skill>" for skill in skills)
    xml = compact_xml(f"<user><name>{escape(name)}</name><age>{age}</age><skills>{skill_nodes}</skills></user>")
    instruction = (
        f"Return XML for a user named {name}, age {age}, with skills {', '.join(skills[:-1])} and {skills[-1]}. "
        f"{pick(rng, XML_PROMPT_SUFFIXES)}"
    )
    return instruction, xml


def xml_product(rng: random.Random) -> tuple[str, str]:
    name = pick(rng, PRODUCTS)
    price = rng.randint(25, 1800)
    in_stock = str(rng.choice([True, False])).lower()
    xml = compact_xml(f"<product><name>{escape(name)}</name><price>{price}</price><in_stock>{in_stock}</in_stock></product>")
    instruction = f"Return XML for product {name} with price {price} and in_stock {in_stock}. {pick(rng, XML_PROMPT_SUFFIXES)}"
    return instruction, xml


def xml_status(rng: random.Random) -> tuple[str, str]:
    job_id = f"job_{rng.randint(100000, 999999)}"
    xml = compact_xml(
        f'<job id="{job_id}">'
        f"<status>{pick(rng, STATUSES)}</status>"
        f"<progress>{rng.randint(0, 100)}</progress>"
        f"<retryable>{str(rng.choice([True, False])).lower()}</retryable>"
        "</job>"
    )
    instruction = f"Return status for {job_id} as XML with a job root element. {pick(rng, XML_PROMPT_SUFFIXES)}"
    return instruction, xml


JSON_BUILDERS = [json_user_custom, json_user_schema, json_product, json_status, json_order]
XML_BUILDERS = [xml_user_custom, xml_product, xml_status]


def make_example(rng: random.Random, formats: list[str]) -> dict[str, str]:
    fmt = pick(rng, formats)
    if fmt == "json":
        instruction, response = pick(rng, JSON_BUILDERS)(rng)
    else:
        instruction, response = pick(rng, XML_BUILDERS)(rng)
    return {"instruction": instruction, "input": "", "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict JSON/XML boundary SFT arrays.")
    parser.add_argument("--tokenizer", default="tokenizer_fineweb_16k/tokenizer.json")
    parser.add_argument("--out_dir", default="data/sft/structured_strict_1024")
    parser.add_argument("--max_seq_len", type=positive_int, default=1024)
    parser.add_argument("--train_examples", type=positive_int, default=120_000)
    parser.add_argument("--val_examples", type=positive_int, default=3_000)
    parser.add_argument("--formats", type=parse_formats, default=["json", "xml"], help="json, xml, or json,xml")
    parser.add_argument("--seed", type=int, default=2026)
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
        "format": "synthetic_structured_strict_outputs",
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
