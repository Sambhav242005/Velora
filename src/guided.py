from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import sre_parse


@dataclass(frozen=True)
class CharPredicate:
    literals: frozenset[str] = frozenset()
    ranges: tuple[tuple[str, str], ...] = ()
    categories: tuple[Any, ...] = ()
    negated: bool = False
    any_char: bool = False

    def matches(self, char: str) -> bool:
        matched = self.any_char and char != "\n"
        if char in self.literals:
            matched = True
        if not matched:
            codepoint = ord(char)
            matched = any(ord(start) <= codepoint <= ord(end) for start, end in self.ranges)
        if not matched:
            matched = any(_category_matches(category, char) for category in self.categories)
        return not matched if self.negated else matched


class RegexNFA:
    """Small Thompson-NFA compiler for common Python regex constructs."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.epsilon_edges: list[set[int]] = []
        self.char_edges: list[list[tuple[CharPredicate, int]]] = []
        parsed = sre_parse.parse(pattern)
        self.start_state, self.accept_state = self._compile_sequence(list(parsed))
        self.start_states = self.epsilon_closure(frozenset({self.start_state}))

    def _new_state(self) -> int:
        state = len(self.epsilon_edges)
        self.epsilon_edges.append(set())
        self.char_edges.append([])
        return state

    def _add_epsilon(self, source: int, target: int) -> None:
        self.epsilon_edges[source].add(target)

    def _add_char_edge(self, source: int, predicate: CharPredicate, target: int) -> None:
        self.char_edges[source].append((predicate, target))

    def _epsilon_fragment(self) -> tuple[int, int]:
        start = self._new_state()
        end = self._new_state()
        self._add_epsilon(start, end)
        return start, end

    def _char_fragment(self, predicate: CharPredicate) -> tuple[int, int]:
        start = self._new_state()
        end = self._new_state()
        self._add_char_edge(start, predicate, end)
        return start, end

    def _compile_sequence(self, tokens: Sequence[tuple[Any, Any]]) -> tuple[int, int]:
        if not tokens:
            return self._epsilon_fragment()

        start = None
        previous_end = None
        for op, arg in tokens:
            fragment_start, fragment_end = self._compile_token(op, arg)
            if start is None:
                start = fragment_start
            else:
                self._add_epsilon(previous_end, fragment_start)
            previous_end = fragment_end
        return int(start), int(previous_end)

    def _compile_token(self, op: Any, arg: Any) -> tuple[int, int]:
        if op is sre_parse.LITERAL:
            return self._char_fragment(CharPredicate(literals=frozenset({chr(arg)})))
        if op is sre_parse.NOT_LITERAL:
            return self._char_fragment(CharPredicate(literals=frozenset({chr(arg)}), negated=True))
        if op is sre_parse.IN:
            return self._char_fragment(_predicate_from_in(arg))
        if op is sre_parse.ANY:
            return self._char_fragment(CharPredicate(any_char=True))
        if op is sre_parse.CATEGORY:
            return self._char_fragment(CharPredicate(categories=(arg,)))
        if op is sre_parse.SUBPATTERN:
            _, _, _, subpattern = arg
            return self._compile_sequence(list(subpattern))
        if op is sre_parse.BRANCH:
            _, branches = arg
            return self._compile_branch(branches)
        if op in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            min_count, max_count, subpattern = arg
            return self._compile_repeat(int(min_count), max_count, list(subpattern))
        if op is sre_parse.AT:
            return self._epsilon_fragment()
        raise ValueError(f"Unsupported regex construct {op!r} in pattern {self.pattern!r}")

    def _compile_branch(self, branches: Sequence[Sequence[tuple[Any, Any]]]) -> tuple[int, int]:
        start = self._new_state()
        end = self._new_state()
        for branch in branches:
            branch_start, branch_end = self._compile_sequence(list(branch))
            self._add_epsilon(start, branch_start)
            self._add_epsilon(branch_end, end)
        return start, end

    def _compile_repeat(self, min_count: int, max_count: Any, tokens: Sequence[tuple[Any, Any]]) -> tuple[int, int]:
        start = self._new_state()
        end = self._new_state()
        current = start

        for _ in range(min_count):
            fragment_start, fragment_end = self._compile_sequence(tokens)
            self._add_epsilon(current, fragment_start)
            current = fragment_end

        if max_count == min_count:
            self._add_epsilon(current, end)
            return start, end

        if max_count is sre_parse.MAXREPEAT:
            self._add_epsilon(current, end)
            fragment_start, fragment_end = self._compile_sequence(tokens)
            self._add_epsilon(current, fragment_start)
            self._add_epsilon(fragment_end, fragment_start)
            self._add_epsilon(fragment_end, end)
            return start, end

        for _ in range(int(max_count) - min_count):
            self._add_epsilon(current, end)
            fragment_start, fragment_end = self._compile_sequence(tokens)
            self._add_epsilon(current, fragment_start)
            current = fragment_end
        self._add_epsilon(current, end)
        return start, end

    def epsilon_closure(self, states: frozenset[int]) -> frozenset[int]:
        closed = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for target in self.epsilon_edges[state]:
                if target not in closed:
                    closed.add(target)
                    stack.append(target)
        return frozenset(closed)

    def advance(self, states: frozenset[int], text: str) -> frozenset[int]:
        current = states
        for char in text:
            next_states: set[int] = set()
            for state in current:
                for predicate, target in self.char_edges[state]:
                    if predicate.matches(char):
                        next_states.add(target)
            if not next_states:
                return frozenset()
            current = self.epsilon_closure(frozenset(next_states))
        return current

    def is_accepting(self, states: frozenset[int]) -> bool:
        return self.accept_state in states


class IndexedRegexLogitsProcessor:
    """Paper-style guided decoding: automaton state -> indexed token transitions.

    The first time a DFA state-set is reached, the tokenizer vocabulary is scanned
    once to build valid token IDs and token -> next-state transitions. During
    generation, the current automaton state is advanced incrementally and logits
    are masked from the cached transition index.
    """

    def __init__(
        self,
        tokenizer,
        pattern: str,
        prompt_lengths: Sequence[int],
        eos_token_id: int | None = None,
        unk_text: str | None = None,
        disallow_special: bool = True,
    ) -> None:
        self.automaton = RegexNFA(pattern)
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

        self._index_cache: dict[frozenset[int], tuple[list[int], dict[int, frozenset[int]]]] = {}
        self._batch_states = [self.automaton.start_states for _ in self.prompt_lengths]
        self._consumed_lengths = list(self.prompt_lengths)

    def _ensure_batch(self, batch_size: int) -> None:
        while len(self._batch_states) < batch_size:
            self._batch_states.append(self.automaton.start_states)
            self._consumed_lengths.append(self.prompt_lengths[-1])

    def _index_for_state(self, state: frozenset[int]) -> tuple[list[int], dict[int, frozenset[int]]]:
        cached = self._index_cache.get(state)
        if cached is not None:
            return cached

        allowed: list[int] = []
        transitions: dict[int, frozenset[int]] = {}
        if self.automaton.is_accepting(state) and self.eos_token_id is not None:
            allowed.append(int(self.eos_token_id))

        for token_id, token_text in enumerate(self.token_texts):
            if token_id == self.eos_token_id or token_id in self.special_token_ids or token_text == "":
                continue
            next_state = self.automaton.advance(state, token_text)
            if next_state:
                allowed.append(token_id)
                transitions[token_id] = next_state

        self._index_cache[state] = (allowed, transitions)
        return allowed, transitions

    def _consume_new_tokens(self, batch_idx: int, idx: torch.Tensor) -> frozenset[int]:
        prompt_length = self.prompt_lengths[min(batch_idx, len(self.prompt_lengths) - 1)]
        consumed = self._consumed_lengths[batch_idx]
        state = self._batch_states[batch_idx]

        if consumed < prompt_length or consumed > idx.size(0):
            consumed = prompt_length
            state = self.automaton.start_states

        for token_id in idx[consumed:].tolist():
            _, transitions = self._index_for_state(state)
            next_state = transitions.get(int(token_id))
            if next_state is None:
                token_text = self.token_texts[int(token_id)] if int(token_id) < self.vocab_size else f"<id:{token_id}>"
                raise RuntimeError(f"Generated token {token_text!r} is invalid for the regex guide.")
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
                raise RuntimeError("Regex guide has no valid next token. Check the regex or use a shorter max_new_tokens.")

            allowed_ids = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            row = logits[batch_idx].clone()
            logits[batch_idx].fill_(-float("inf"))
            logits[batch_idx, allowed_ids] = row[allowed_ids]


# Backward-compatible name for imports/tests that used the first implementation.
PrefixRegexLogitsProcessor = IndexedRegexLogitsProcessor


def _predicate_from_in(items: Sequence[tuple[Any, Any]]) -> CharPredicate:
    literals: set[str] = set()
    ranges: list[tuple[str, str]] = []
    categories: list[Any] = []
    negated = False
    for op, arg in items:
        if op is sre_parse.NEGATE:
            negated = True
        elif op is sre_parse.LITERAL:
            literals.add(chr(arg))
        elif op is sre_parse.RANGE:
            start, end = arg
            ranges.append((chr(start), chr(end)))
        elif op is sre_parse.CATEGORY:
            categories.append(arg)
        else:
            raise ValueError(f"Unsupported character class construct {op!r}")
    return CharPredicate(
        literals=frozenset(literals),
        ranges=tuple(ranges),
        categories=tuple(categories),
        negated=negated,
    )


def _category_matches(category: Any, char: str) -> bool:
    if category is sre_parse.CATEGORY_SPACE:
        return char.isspace()
    if category is sre_parse.CATEGORY_NOT_SPACE:
        return not char.isspace()
    if category is sre_parse.CATEGORY_DIGIT:
        return char.isdigit()
    if category is sre_parse.CATEGORY_NOT_DIGIT:
        return not char.isdigit()
    if category is sre_parse.CATEGORY_WORD:
        return char == "_" or char.isalnum()
    if category is sre_parse.CATEGORY_NOT_WORD:
        return not (char == "_" or char.isalnum())
    if category is sre_parse.CATEGORY_LINEBREAK:
        return char == "\n"
    if category is sre_parse.CATEGORY_NOT_LINEBREAK:
        return char != "\n"
    raise ValueError(f"Unsupported regex category {category!r}")


def add_regex_guidance_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--regex",
        default=None,
        help="Constrain the generated suffix to this regex using an indexed automaton over the tokenizer vocab.",
    )
    parser.add_argument(
        "--regex_unk_text",
        default=None,
        help="Treat the tokenizer <unk> token as this literal text while guiding, e.g. '{' for JSON.",
    )
    parser.add_argument(
        "--regex_no_eos",
        action="store_true",
        help="Do not force <eos> once the generated suffix fully matches --regex.",
    )


def build_regex_logits_processor(args: Namespace, tokenizer, prompt_lengths: Sequence[int]):
    pattern = getattr(args, "regex", None)
    if not pattern:
        return None, None
    eos_token_id = None if getattr(args, "regex_no_eos", False) else tokenizer.token_to_id("<eos>")
    processor = IndexedRegexLogitsProcessor(
        tokenizer=tokenizer,
        pattern=pattern,
        prompt_lengths=prompt_lengths,
        eos_token_id=eos_token_id,
        unk_text=getattr(args, "regex_unk_text", None),
    )
    return processor, eos_token_id
