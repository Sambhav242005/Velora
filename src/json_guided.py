from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal, Sequence
import re

import torch

_HEX = set("0123456789abcdefABCDEF")
_WS = {" ", "\t", "\n", "\r"}


@dataclass(frozen=True)
class JsonState:
    stack: tuple[str, ...]
    mode: str = "DEFAULT"
    literal: str = ""
    literal_pos: int = 0
    number_state: str = ""
    string_after: str = ""
    unicode_remaining: int = 0


class JsonRecognizer:
    """Incremental JSON recognizer used for grammar-guided decoding.

    The recognizer is deterministic and character based. The logits processor
    indexes tokenizer tokens against these states, which is the same practical
    shape as the paper's parser-state -> vocabulary subset map.
    """

    def __init__(self, root: Literal["object", "value"] = "object") -> None:
        if root not in {"object", "value"}:
            raise ValueError("root must be 'object' or 'value'")
        self.start_state = JsonState(stack=("ROOT_OBJECT" if root == "object" else "ROOT_VALUE",))

    def advance(self, state: JsonState, text: str) -> JsonState | None:
        current: JsonState | None = state
        for char in text:
            current = self._step(current, char) if current is not None else None
            if current is None:
                return None
        return current

    def is_accepting(self, state: JsonState) -> bool:
        if state.mode == "DEFAULT":
            return state.stack == ("END",)
        if state.mode == "NUMBER" and self._can_end_number(state.number_state):
            completed = self._complete_value(JsonState(stack=state.stack))
            return completed is not None and completed.stack == ("END",)
        return False

    def _step(self, state: JsonState, char: str) -> JsonState | None:
        current = state
        reprocess = True
        while reprocess:
            reprocess = False
            if current.mode == "STRING":
                return self._step_string(current, char)
            if current.mode == "STRING_ESCAPE":
                return self._step_string_escape(current, char)
            if current.mode == "STRING_UNICODE":
                return self._step_string_unicode(current, char)
            if current.mode == "LITERAL":
                return self._step_literal(current, char)
            if current.mode == "NUMBER":
                next_state, should_reprocess = self._step_number(current, char)
                if next_state is None:
                    return None
                current = next_state
                reprocess = should_reprocess
                continue
            return self._step_default(current, char)
        return current

    def _step_default(self, state: JsonState, char: str) -> JsonState | None:
        if not state.stack:
            return None
        top = state.stack[-1]
        if top == "END":
            return state if char in _WS else None
        if char in _WS:
            return state

        if top == "ROOT_OBJECT":
            return self._start_object(state, char) if char == "{" else None
        if top == "ROOT_VALUE":
            return self._start_value(state, char)
        if top == "OBJ_KEY_OR_END":
            if char == "}":
                return self._close_container(state)
            if char == '"':
                return JsonState(stack=state.stack, mode="STRING", string_after="KEY")
            return None
        if top == "OBJ_KEY":
            return JsonState(stack=state.stack, mode="STRING", string_after="KEY") if char == '"' else None
        if top == "OBJ_COLON":
            if char == ":":
                return JsonState(stack=state.stack[:-1] + ("OBJ_VALUE",))
            return None
        if top == "OBJ_VALUE":
            return self._start_value(state, char)
        if top == "OBJ_COMMA_OR_END":
            if char == ",":
                return JsonState(stack=state.stack[:-1] + ("OBJ_KEY",))
            if char == "}":
                return self._close_container(state)
            return None
        if top == "ARR_VALUE_OR_END":
            if char == "]":
                return self._close_container(state)
            return self._start_value(state, char)
        if top == "ARR_VALUE":
            return self._start_value(state, char)
        if top == "ARR_COMMA_OR_END":
            if char == ",":
                return JsonState(stack=state.stack[:-1] + ("ARR_VALUE",))
            if char == "]":
                return self._close_container(state)
            return None
        return None

    def _start_value(self, state: JsonState, char: str) -> JsonState | None:
        if char == "{":
            return self._start_object(state, char)
        if char == "[":
            return JsonState(stack=state.stack + ("ARR_VALUE_OR_END",))
        if char == '"':
            return JsonState(stack=state.stack, mode="STRING", string_after="VALUE")
        if char == "t":
            return JsonState(stack=state.stack, mode="LITERAL", literal="true", literal_pos=1)
        if char == "f":
            return JsonState(stack=state.stack, mode="LITERAL", literal="false", literal_pos=1)
        if char == "n":
            return JsonState(stack=state.stack, mode="LITERAL", literal="null", literal_pos=1)
        number_state = self._start_number_state(char)
        if number_state is not None:
            return JsonState(stack=state.stack, mode="NUMBER", number_state=number_state)
        return None

    def _start_object(self, state: JsonState, char: str) -> JsonState | None:
        if char != "{":
            return None
        return JsonState(stack=state.stack + ("OBJ_KEY_OR_END",))

    def _close_container(self, state: JsonState) -> JsonState | None:
        if len(state.stack) < 2:
            return None
        return self._complete_value(JsonState(stack=state.stack[:-1]))

    def _complete_value(self, state: JsonState) -> JsonState | None:
        if not state.stack:
            return None
        top = state.stack[-1]
        if top in {"ROOT_OBJECT", "ROOT_VALUE"}:
            return JsonState(stack=("END",))
        if top in {"OBJ_VALUE"}:
            return JsonState(stack=state.stack[:-1] + ("OBJ_COMMA_OR_END",))
        if top in {"ARR_VALUE", "ARR_VALUE_OR_END"}:
            return JsonState(stack=state.stack[:-1] + ("ARR_COMMA_OR_END",))
        return None

    def _step_string(self, state: JsonState, char: str) -> JsonState | None:
        if char == '"':
            if state.string_after == "KEY":
                return JsonState(stack=state.stack[:-1] + ("OBJ_COLON",))
            if state.string_after == "VALUE":
                return self._complete_value(JsonState(stack=state.stack))
            return None
        if char == "\\":
            return JsonState(stack=state.stack, mode="STRING_ESCAPE", string_after=state.string_after)
        if ord(char) < 0x20:
            return None
        return state

    def _step_string_escape(self, state: JsonState, char: str) -> JsonState | None:
        if char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            return JsonState(stack=state.stack, mode="STRING", string_after=state.string_after)
        if char == "u":
            return JsonState(
                stack=state.stack,
                mode="STRING_UNICODE",
                string_after=state.string_after,
                unicode_remaining=4,
            )
        return None

    def _step_string_unicode(self, state: JsonState, char: str) -> JsonState | None:
        if char not in _HEX:
            return None
        remaining = state.unicode_remaining - 1
        if remaining <= 0:
            return JsonState(stack=state.stack, mode="STRING", string_after=state.string_after)
        return JsonState(
            stack=state.stack,
            mode="STRING_UNICODE",
            string_after=state.string_after,
            unicode_remaining=remaining,
        )

    def _step_literal(self, state: JsonState, char: str) -> JsonState | None:
        if state.literal_pos >= len(state.literal) or char != state.literal[state.literal_pos]:
            return None
        pos = state.literal_pos + 1
        if pos == len(state.literal):
            return self._complete_value(JsonState(stack=state.stack))
        return JsonState(stack=state.stack, mode="LITERAL", literal=state.literal, literal_pos=pos)

    def _start_number_state(self, char: str) -> str | None:
        if char == "-":
            return "AFTER_MINUS"
        if char == "0":
            return "ZERO"
        if "1" <= char <= "9":
            return "INT"
        return None

    def _step_number(self, state: JsonState, char: str) -> tuple[JsonState | None, bool]:
        number_state = state.number_state
        next_number_state: str | None = None
        if number_state == "AFTER_MINUS":
            if char == "0":
                next_number_state = "ZERO"
            elif "1" <= char <= "9":
                next_number_state = "INT"
        elif number_state == "ZERO":
            if char == ".":
                next_number_state = "DOT"
            elif char in {"e", "E"}:
                next_number_state = "EXP"
        elif number_state == "INT":
            if char.isdigit():
                next_number_state = "INT"
            elif char == ".":
                next_number_state = "DOT"
            elif char in {"e", "E"}:
                next_number_state = "EXP"
        elif number_state == "DOT":
            if char.isdigit():
                next_number_state = "FRAC"
        elif number_state == "FRAC":
            if char.isdigit():
                next_number_state = "FRAC"
            elif char in {"e", "E"}:
                next_number_state = "EXP"
        elif number_state == "EXP":
            if char in {"+", "-"}:
                next_number_state = "EXP_SIGN"
            elif char.isdigit():
                next_number_state = "EXP_DIGITS"
        elif number_state == "EXP_SIGN":
            if char.isdigit():
                next_number_state = "EXP_DIGITS"
        elif number_state == "EXP_DIGITS":
            if char.isdigit():
                next_number_state = "EXP_DIGITS"

        if next_number_state is not None:
            return JsonState(stack=state.stack, mode="NUMBER", number_state=next_number_state), False
        if self._can_end_number(number_state):
            completed = self._complete_value(JsonState(stack=state.stack))
            return completed, True
        return None, False

    def _can_end_number(self, number_state: str) -> bool:
        return number_state in {"ZERO", "INT", "FRAC", "EXP_DIGITS"}


class JsonLogitsProcessor:
    """JSON grammar guide with cached parser-state -> token transitions."""

    def __init__(
        self,
        tokenizer,
        prompt_lengths: Sequence[int],
        eos_token_id: int | None = None,
        root: Literal["object", "value"] = "object",
        unk_text: str | None = "{",
        disallow_special: bool = True,
    ) -> None:
        self.recognizer = JsonRecognizer(root=root)
        self.prompt_lengths = [int(length) for length in prompt_lengths]
        self.eos_token_id = eos_token_id
        self.vocab_size = int(tokenizer.get_vocab_size())
        self.token_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in range(self.vocab_size)
        ]
        self.special_token_ids: set[int] = set()
        if disallow_special:
            for token in ("<pad>", "<bos>", "<eos>", "<unk>"):
                token_id = tokenizer.token_to_id(token)
                if token_id is not None:
                    self.special_token_ids.add(int(token_id))

        if eos_token_id is not None:
            self.special_token_ids.discard(int(eos_token_id))
        if unk_text is not None:
            unk_id = tokenizer.token_to_id("<unk>")
            if unk_id is not None:
                self.token_texts[int(unk_id)] = unk_text
                self.special_token_ids.discard(int(unk_id))

        self._index_cache: dict[JsonState, tuple[list[int], dict[int, JsonState]]] = {}
        self._batch_states = [self.recognizer.start_state for _ in self.prompt_lengths]
        self._consumed_lengths = list(self.prompt_lengths)

    def _ensure_batch(self, batch_size: int) -> None:
        while len(self._batch_states) < batch_size:
            self._batch_states.append(self.recognizer.start_state)
            self._consumed_lengths.append(self.prompt_lengths[-1])

    def _index_for_state(self, state: JsonState) -> tuple[list[int], dict[int, JsonState]]:
        cached = self._index_cache.get(state)
        if cached is not None:
            return cached

        allowed: list[int] = []
        transitions: dict[int, JsonState] = {}
        if self.recognizer.is_accepting(state) and self.eos_token_id is not None:
            allowed.append(int(self.eos_token_id))

        for token_id, token_text in enumerate(self.token_texts):
            if token_id == self.eos_token_id or token_id in self.special_token_ids or token_text == "":
                continue
            next_state = self.recognizer.advance(state, token_text)
            if next_state is not None:
                allowed.append(token_id)
                transitions[token_id] = next_state

        self._index_cache[state] = (allowed, transitions)
        return allowed, transitions

    def _consume_new_tokens(self, batch_idx: int, idx: torch.Tensor) -> JsonState:
        prompt_length = self.prompt_lengths[min(batch_idx, len(self.prompt_lengths) - 1)]
        consumed = self._consumed_lengths[batch_idx]
        state = self._batch_states[batch_idx]

        if consumed < prompt_length or consumed > idx.size(0):
            consumed = prompt_length
            state = self.recognizer.start_state

        for token_id in idx[consumed:].tolist():
            _, transitions = self._index_for_state(state)
            next_state = transitions.get(int(token_id))
            if next_state is None:
                token_text = self.token_texts[int(token_id)] if int(token_id) < self.vocab_size else f"<id:{token_id}>"
                raise RuntimeError(f"Generated token {token_text!r} is invalid for the JSON guide.")
            state = next_state
            consumed += 1

        self._batch_states[batch_idx] = state
        self._consumed_lengths[batch_idx] = consumed
        return state

    def __call__(self, logits: torch.Tensor, idx: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, V], got {tuple(logits.shape)}")
        if logits.size(1) != self.vocab_size:
            raise ValueError(f"Logits vocab size {logits.size(1)} != tokenizer vocab size {self.vocab_size}")
        if idx.size(0) != logits.size(0):
            raise ValueError(f"Batch mismatch: idx batch {idx.size(0)} != logits batch {logits.size(0)}")

        self._ensure_batch(logits.size(0))
        for batch_idx in range(logits.size(0)):
            state = self._consume_new_tokens(batch_idx, idx[batch_idx])
            allowed, _ = self._index_for_state(state)
            if not allowed:
                raise RuntimeError("JSON guide has no valid next token. Increase max_new_tokens or relax the guide.")

            allowed_ids = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            row = logits[batch_idx].clone()
            logits[batch_idx].fill_(-float("inf"))
            logits[batch_idx, allowed_ids] = row[allowed_ids]


@dataclass(frozen=True)
class FixedJsonObjectState:
    key_index: int = 0
    phase: str = "START"
    key_pos: int = 0
    value_state: JsonState | None = None


class FixedJsonObjectRecognizer:
    """Incremental recognizer for an object with a fixed ordered key set."""

    def __init__(self, keys: Sequence[str], key_types: dict[str, str] | None = None) -> None:
        if not keys:
            raise ValueError("FixedJsonObjectRecognizer requires at least one key.")
        self.keys = tuple(keys)
        self.key_types = key_types or {}
        self.value_recognizer = JsonRecognizer(root="value")
        self.start_state = FixedJsonObjectState()

    def advance(self, state: FixedJsonObjectState, text: str) -> FixedJsonObjectState | None:
        current: FixedJsonObjectState | None = state
        for char in text:
            current = self._step(current, char) if current is not None else None
            if current is None:
                return None
        return current

    def is_accepting(self, state: FixedJsonObjectState) -> bool:
        return state.phase == "END"

    def _char_allowed_for_value_type(self, key: str, char: str) -> bool:
        value_type = self.key_types.get(key, "any")
        if value_type == "any":
            return True
        if value_type == "string":
            return char == '"'
        if value_type == "boolean":
            return char in {"t", "f"}
        if value_type == "number":
            return char == "-" or char.isdigit()
        if value_type == "object":
            return char == "{"
        if value_type == "array":
            return char == "["
        if value_type == "null":
            return char == "n"
        return True

    def _step(self, state: FixedJsonObjectState, char: str) -> FixedJsonObjectState | None:
        current = state
        reprocess = True
        while reprocess:
            reprocess = False
            phase = current.phase
            if phase == "START":
                if char in _WS:
                    return current
                if char == "{":
                    return FixedJsonObjectState(key_index=0, phase="BEFORE_KEY")
                return None
            if phase == "BEFORE_KEY":
                if char in _WS:
                    return current
                if char == '"':
                    return FixedJsonObjectState(key_index=current.key_index, phase="KEY", key_pos=0)
                return None
            if phase == "KEY":
                key = self.keys[current.key_index]
                if current.key_pos < len(key) and char == key[current.key_pos]:
                    return FixedJsonObjectState(
                        key_index=current.key_index,
                        phase="KEY",
                        key_pos=current.key_pos + 1,
                    )
                if current.key_pos == len(key) and char == '"':
                    return FixedJsonObjectState(key_index=current.key_index, phase="AFTER_KEY")
                return None
            if phase == "AFTER_KEY":
                if char in _WS:
                    return current
                if char == ":":
                    return FixedJsonObjectState(key_index=current.key_index, phase="BEFORE_VALUE")
                return None
            if phase == "BEFORE_VALUE":
                if char in _WS:
                    return current
                key = self.keys[current.key_index]
                if not self._char_allowed_for_value_type(key, char):
                    return None
                value_state = self.value_recognizer.advance(self.value_recognizer.start_state, char)
                if value_state is None:
                    return None
                return FixedJsonObjectState(
                    key_index=current.key_index,
                    phase="VALUE",
                    value_state=value_state,
                )
            if phase == "VALUE":
                assert current.value_state is not None
                next_value_state = self.value_recognizer.advance(current.value_state, char)
                if next_value_state is not None:
                    return FixedJsonObjectState(
                        key_index=current.key_index,
                        phase="VALUE",
                        value_state=next_value_state,
                    )
                if self.value_recognizer.is_accepting(current.value_state):
                    current = FixedJsonObjectState(key_index=current.key_index, phase="AFTER_VALUE")
                    reprocess = True
                    continue
                return None
            if phase == "AFTER_VALUE":
                if char in _WS:
                    return current
                is_last = current.key_index == len(self.keys) - 1
                if not is_last and char == ",":
                    return FixedJsonObjectState(key_index=current.key_index + 1, phase="BEFORE_KEY")
                if is_last and char == "}":
                    return FixedJsonObjectState(key_index=current.key_index, phase="END")
                return None
            if phase == "END":
                return current if char in _WS else None
        return current


class FixedJsonObjectLogitsProcessor:
    """JSON schema-lite guide for exact root-object keys."""

    def __init__(
        self,
        tokenizer,
        keys: Sequence[str],
        prompt_lengths: Sequence[int],
        eos_token_id: int | None = None,
        key_types: dict[str, str] | None = None,
        unk_text: str | None = "{",
        disallow_special: bool = True,
    ) -> None:
        self.recognizer = FixedJsonObjectRecognizer(keys, key_types=key_types)
        self.prompt_lengths = [int(length) for length in prompt_lengths]
        self.eos_token_id = eos_token_id
        self.vocab_size = int(tokenizer.get_vocab_size())
        self.token_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in range(self.vocab_size)
        ]
        self.special_token_ids: set[int] = set()
        if disallow_special:
            for token in ("<pad>", "<bos>", "<eos>", "<unk>"):
                token_id = tokenizer.token_to_id(token)
                if token_id is not None:
                    self.special_token_ids.add(int(token_id))

        if eos_token_id is not None:
            self.special_token_ids.discard(int(eos_token_id))
        if unk_text is not None:
            unk_id = tokenizer.token_to_id("<unk>")
            if unk_id is not None:
                self.token_texts[int(unk_id)] = unk_text
                self.special_token_ids.discard(int(unk_id))

        self._index_cache: dict[FixedJsonObjectState, tuple[list[int], dict[int, FixedJsonObjectState]]] = {}
        self._batch_states = [self.recognizer.start_state for _ in self.prompt_lengths]
        self._consumed_lengths = list(self.prompt_lengths)

    def _ensure_batch(self, batch_size: int) -> None:
        while len(self._batch_states) < batch_size:
            self._batch_states.append(self.recognizer.start_state)
            self._consumed_lengths.append(self.prompt_lengths[-1])

    def _index_for_state(self, state: FixedJsonObjectState) -> tuple[list[int], dict[int, FixedJsonObjectState]]:
        cached = self._index_cache.get(state)
        if cached is not None:
            return cached

        allowed: list[int] = []
        transitions: dict[int, FixedJsonObjectState] = {}
        if self.recognizer.is_accepting(state) and self.eos_token_id is not None:
            allowed.append(int(self.eos_token_id))

        for token_id, token_text in enumerate(self.token_texts):
            if token_id == self.eos_token_id or token_id in self.special_token_ids or token_text == "":
                continue
            next_state = self.recognizer.advance(state, token_text)
            if next_state is not None:
                allowed.append(token_id)
                transitions[token_id] = next_state

        self._index_cache[state] = (allowed, transitions)
        return allowed, transitions

    def _consume_new_tokens(self, batch_idx: int, idx: torch.Tensor) -> FixedJsonObjectState:
        prompt_length = self.prompt_lengths[min(batch_idx, len(self.prompt_lengths) - 1)]
        consumed = self._consumed_lengths[batch_idx]
        state = self._batch_states[batch_idx]

        if consumed < prompt_length or consumed > idx.size(0):
            consumed = prompt_length
            state = self.recognizer.start_state

        for token_id in idx[consumed:].tolist():
            _, transitions = self._index_for_state(state)
            next_state = transitions.get(int(token_id))
            if next_state is None:
                token_text = self.token_texts[int(token_id)] if int(token_id) < self.vocab_size else f"<id:{token_id}>"
                raise RuntimeError(f"Generated token {token_text!r} is invalid for the fixed-key JSON guide.")
            state = next_state
            consumed += 1

        self._batch_states[batch_idx] = state
        self._consumed_lengths[batch_idx] = consumed
        return state

    def __call__(self, logits: torch.Tensor, idx: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, V], got {tuple(logits.shape)}")
        if logits.size(1) != self.vocab_size:
            raise ValueError(f"Logits vocab size {logits.size(1)} != tokenizer vocab size {self.vocab_size}")
        if idx.size(0) != logits.size(0):
            raise ValueError(f"Batch mismatch: idx batch {idx.size(0)} != logits batch {logits.size(0)}")

        self._ensure_batch(logits.size(0))
        for batch_idx in range(logits.size(0)):
            state = self._consume_new_tokens(batch_idx, idx[batch_idx])
            allowed, _ = self._index_for_state(state)
            if not allowed:
                raise RuntimeError("Fixed-key JSON guide has no valid next token.")

            allowed_ids = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            row = logits[batch_idx].clone()
            logits[batch_idx].fill_(-float("inf"))
            logits[batch_idx, allowed_ids] = row[allowed_ids]


def infer_json_keys(text: str) -> list[str]:
    match = re.search(r"\bkeys?\s+([^.;:\n]+)", text, flags=re.IGNORECASE)
    if match is None:
        return []
    segment = match.group(1)
    quoted = re.findall(r"[`'\"]([A-Za-z_][A-Za-z0-9_-]*)[`'\"]", segment)
    if quoted:
        return quoted
    parts = [part.strip() for part in re.split(r"\s*(?:,|\band\b)\s*", segment) if part.strip()]
    keys = []
    for part in parts:
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", part, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"[^A-Za-z0-9_-].*$", "", cleaned)
        if cleaned and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", cleaned):
            keys.append(cleaned)
    return keys


def add_json_guidance_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--no_json_guide",
        action="store_true",
        help="Disable automatic JSON grammar guidance for --format json.",
    )
    parser.add_argument(
        "--json_root",
        choices=["object", "value"],
        default="object",
        help="Root JSON grammar to enforce when --format json is guided and no fixed keys are provided.",
    )
    parser.add_argument(
        "--json_keys",
        default=None,
        help="Comma-separated root-object keys to force, e.g. answer,confidence. If omitted, generate_structured.py tries to infer keys from the instruction.",
    )
    parser.add_argument(
        "--json_key_types",
        default=None,
        help="Comma-separated key:type constraints, e.g. answer:boolean,confidence:number. Supported types: string, boolean, number, object, array, null, any.",
    )


_SUPPORTED_KEY_TYPES = {"string", "boolean", "number", "object", "array", "null", "any"}


def _parse_key_types(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    result: dict[str, str] = {}
    for part in raw.split(","):
        if ":" not in part:
            raise ValueError("--json_key_types entries must look like key:type")
        key, value_type = [item.strip() for item in part.split(":", 1)]
        if value_type not in _SUPPORTED_KEY_TYPES:
            raise ValueError(f"Unsupported JSON key type {value_type!r}")
        if key:
            result[key] = value_type
    return result


def _infer_key_types(keys: Sequence[str], instruction: str) -> dict[str, str]:
    lowered_instruction = instruction.lower()
    result: dict[str, str] = {}
    for key in keys:
        lowered_key = key.lower()
        if lowered_key in {"confidence", "score", "probability", "prob"}:
            result[key] = "number"
        elif lowered_key in {"answer", "correct", "is_correct", "result"} and (
            "whether" in lowered_instruction or "yes or no" in lowered_instruction or "true or false" in lowered_instruction
        ):
            result[key] = "boolean"
    return result


def build_json_logits_processor(args: Namespace, tokenizer, prompt_lengths: Sequence[int]):
    eos_token_id = tokenizer.token_to_id("<eos>")
    instruction = getattr(args, "instruction", "")
    keys_arg = getattr(args, "json_keys", None)
    keys = [part.strip() for part in keys_arg.split(",") if part.strip()] if keys_arg else []
    if not keys:
        keys = infer_json_keys(instruction)
    key_types = _parse_key_types(getattr(args, "json_key_types", None))
    if keys:
        key_types = _infer_key_types(keys, instruction) | key_types
        processor = FixedJsonObjectLogitsProcessor(
            tokenizer=tokenizer,
            keys=keys,
            prompt_lengths=prompt_lengths,
            eos_token_id=eos_token_id,
            key_types=key_types,
        )
    else:
        processor = JsonLogitsProcessor(
            tokenizer=tokenizer,
            prompt_lengths=prompt_lengths,
            eos_token_id=eos_token_id,
            root=getattr(args, "json_root", "object"),
        )
    return processor, eos_token_id
