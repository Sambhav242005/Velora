from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def extract_xml_document(text: str) -> str | None:
    match = re.search(r"<([A-Za-z_][\w:.-]*)(?:\s[^>]*)?>.*</\1>", text, flags=re.DOTALL)
    return match.group(0) if match else None


def score_json(text: str) -> tuple[float, str]:
    stripped = text.strip()
    candidate = extract_json_object(stripped)
    if candidate is None:
        return 0.0, "missing_json_object"
    try:
        json.loads(candidate)
    except Exception as error:
        return 0.0, f"invalid_json:{error}"
    score = 1.0
    if candidate != stripped:
        score -= 0.35
    if not stripped.startswith("{"):
        score -= 0.25
    if not stripped.endswith("}"):
        score -= 0.25
    return max(score, 0.0), "ok"


def score_xml(text: str) -> tuple[float, str]:
    stripped = text.strip()
    candidate = extract_xml_document(stripped)
    if candidate is None:
        return 0.0, "missing_xml_document"
    try:
        ET.fromstring(candidate)
    except Exception as error:
        return 0.0, f"invalid_xml:{error}"
    score = 1.0
    if candidate != stripped:
        score -= 0.35
    if not stripped.startswith("<"):
        score -= 0.25
    if not stripped.endswith(">"):
        score -= 0.25
    return max(score, 0.0), "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one JSON/XML model output from stdin.")
    parser.add_argument("--format", choices=["json", "xml"], required=True)
    args = parser.parse_args()
    text = sys.stdin.read()
    score, reason = score_json(text) if args.format == "json" else score_xml(text)
    print(json.dumps({"score": score, "reason": reason}, separators=(",", ":")))


if __name__ == "__main__":
    main()
