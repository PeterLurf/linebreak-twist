"""Build fixed-width wrapped prose with per-token geometric labels.

Every sample is a prose chunk greedily wrapped at width k (break iff the next word
would overflow), so the ground-truth linebreak rule is exact:
    newline after word  <=>  col + 1 + len(next_word) > k
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

PROSE_SETS = {"Pile-CC", "OpenWebText2", "Wikipedia (en)", "Books3", "Gutenberg (PG-19)", "BookCorpus2"}


def load_prose_docs(min_chars: int = 2500, max_docs: int = 4000, seed: int = 0) -> list[str]:
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = []
    for row in ds:
        if row["meta"].get("pile_set_name") not in PROSE_SETS:
            continue
        text = " ".join(row["text"].split())
        if len(text) < min_chars:
            continue
        letters = sum(c.isalpha() or c == " " for c in text)
        if letters / len(text) < 0.9:
            continue
        if max(len(w) for w in text.split(" ")) > 18:
            # drop docs with URLs / very long tokens; keeps every word < min width
            text = " ".join(w for w in text.split(" ") if len(w) <= 18)
        docs.append(text)
    random.Random(seed).shuffle(docs)
    return docs[:max_docs]


def greedy_wrap(text: str, k: int) -> str:
    lines, cur = [], ""
    for w in text.split(" "):
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= k:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


@dataclass
class Sample:
    input_ids: np.ndarray  # [T]
    # per-token labels (aligned with input_ids; bos gets -1s)
    col: np.ndarray  # chars in current line through end of this token (0 for newline tokens)
    width: np.ndarray  # target width k
    prev_len: np.ndarray  # length of previous line (-1 on line 0)
    max_prev_len: np.ndarray  # max length of all previous lines (-1 on line 0)
    line_idx: np.ndarray
    is_nl: np.ndarray  # token is a newline
    nl_next: np.ndarray  # next token is a newline
    word_end: np.ndarray  # token ends a word (next char is ' ' or '\n')
    next_word_len: np.ndarray  # length of the next word (-1 if not word_end)
    last_nl_tok: np.ndarray  # index of most recent newline token before t (-1 if none)
    prev_nl_tok: np.ndarray  # index of the newline token before that (-1 if none)
    k: int


def label_sample(tok, wrapped: str, k: int, max_tokens: int) -> Sample | None:
    enc = tok(wrapped, return_offsets_mapping=True, add_special_tokens=True)
    ids = np.array(enc["input_ids"])[:max_tokens]
    offs = enc["offset_mapping"][:max_tokens]
    T = len(ids)
    nl_positions = [m.start() for m in re.finditer("\n", wrapped)]
    line_lens = [len(l) for l in wrapped.split("\n")]
    out = {n: np.full(T, -1, dtype=np.int32) for n in
           ["col", "width", "prev_len", "max_prev_len", "line_idx", "next_word_len", "last_nl_tok", "prev_nl_tok"]}
    is_nl = np.zeros(T, bool); nl_next = np.zeros(T, bool); word_end = np.zeros(T, bool)
    last_nl, prev_nl = -1, -1
    for t in range(T):
        s, e = offs[t]
        if s == e and t == 0:  # bos
            continue
        piece = wrapped[s:e]
        if "\n" in piece and piece.strip() != "":
            return None  # newline merged with a word: tokenizer incompatible with this design
        line_idx = wrapped.count("\n", 0, e if "\n" not in piece else s)
        out["line_idx"][t] = line_idx
        out["width"][t] = k
        out["last_nl_tok"][t] = last_nl
        out["prev_nl_tok"][t] = prev_nl
        if line_idx > 0:
            out["prev_len"][t] = line_lens[line_idx - 1]
            out["max_prev_len"][t] = max(line_lens[:line_idx])
        if piece.strip() == "" and "\n" in piece:
            is_nl[t] = True
            out["col"][t] = 0
            prev_nl, last_nl = last_nl, t
            if t > 0:
                nl_next[t - 1] = True
            continue
        line_start = nl_positions[line_idx - 1] + 1 if line_idx > 0 else 0
        out["col"][t] = e - line_start
        nxt = wrapped[e] if e < len(wrapped) else ""
        if nxt in (" ", "\n"):
            word_end[t] = True
            j = e + 1
            m = re.match(r"[^ \n]*", wrapped[j:])
            out["next_word_len"][t] = len(m.group(0))
    return Sample(ids, out["col"], out["width"], out["prev_len"], out["max_prev_len"], out["line_idx"],
                  is_nl, nl_next, word_end, out["next_word_len"], out["last_nl_tok"], out["prev_nl_tok"], k)


def build_dataset(tok, n_samples: int, widths: tuple[int, int], max_tokens: int, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    docs = load_prose_docs(seed=seed)
    samples, i = [], 0
    while len(samples) < n_samples and i < len(docs):
        k = rng.randint(*widths)
        # use a random window of the doc so samples don't all start at document openings
        words = docs[i].split(" ")
        start = rng.randint(0, max(0, len(words) - 600))
        text = " ".join(words[start:start + 600])
        s = label_sample(tok, greedy_wrap(text, k), k, max_tokens)
        if s is not None:
            samples.append(s)
        i += 1
    return samples
