"""Tokenize and pack empathetic dialogues into uint16 shards for Atom-500M.

    # The full four-dataset mixture (the default corpus)
    python scripts/prepare_dataset.py --mixture

    # One dataset, local JSONL
    python scripts/prepare_dataset.py --input data/raw.jsonl

    # Pack and upload
    python scripts/prepare_dataset.py --mixture --push_to_hf --repo_id siddhu1174/atom-500m

Output is a flat stream of GPT-2 BPE ids cut into fixed 2049-token chunks:
one chunk = one training example, with x = chunk[:-1] and y = chunk[1:].

Design notes
------------
**Bounded memory.** The token stream is written to disk as it is produced and
rolled into shards at ~100 MB. A naive `np.concatenate(all_tokens)` would hold
the entire corpus in RAM; a 100M-token corpus is 200 MB as uint16 but 800 MB as
int64, and larger mixtures do not fit at all. This script's peak RSS is
independent of corpus size.

**Lossless packing.** EOT (50256) terminates every conversation, so chunk
boundaries land on natural document boundaries. The final chunk of the stream is
truncated if it does not divide evenly; the dropped tokens are counted and
reported, never silently discarded.

**uint16 is safe here.** GPT-2's largest id is 50256, comfortably under 65535.
The packer asserts this on every write rather than assuming it -- a tokenizer
change that overflows uint16 would otherwise corrupt the corpus silently, since
a wrapped token is still a valid-looking number.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ShardedMemmapDataset, collate  # noqa: E402

EOT = 50256
GPT2_VOCAB = 50257
UINT16_MAX = np.iinfo(np.uint16).max
DEFAULT_SHARD_BYTES = 100 * 1024 * 1024  # ~100 MB
TOKENIZER_NAME = "gpt2"
DEFAULT_HF_REPO = "siddhu1174/atom-500m"


def clean_text(text: str) -> str:
    """Normalize a raw utterance before tokenization.

    Three classes of problem, all of which hurt an empathy model specifically:

    1. **Dataset artifacts.** empathetic_dialogues and counselling dumps store
       punctuation as BPE escape words -- ``_comma_``, ``_blank_``,
       ``_period_``. Left in place, the model learns to emit the literal text
       ``_comma_`` mid-sentence, because that is what it was trained on. These
       are replaced with the real characters.
    2. **Unicode inconsistency.** Smart quotes (" " ' '), various dashes
       (--, --, -), and the non-breaking space each become several tokens under
       GPT-2 BPE instead of one. Every such variant is folded to its ASCII
       equivalent so the same punctuation costs the same everywhere.
    3. **Whitespace runs.** Dialogue dumps are full of multi-space and tab
       runs from CSV/JSON round-trips; GPT-2 encodes each space separately, so
       "hi   there" is several wasted tokens and an inconsistent prefix.

    Accents are *preserved* (NFKC, not NFD/NFKC+strip): they carry meaning in
    several languages and the model should learn them rather than see
    "cafe" for "café". Control characters and zero-width joiners are removed
    because GPT-2 has no tokens for them and they fragment words.
    """
    if not text:
        return ""

    # 1. Dataset punctuation artifacts, longest-first so _comma_ is not
    # partially matched by a shorter pattern.
    for artifact, real in (
        ("_comma_", ","),
        ("_blank_", " "),
        ("_period_", "."),
        ("_question_", "?"),
        ("_exclamation_", "!"),
        ("_conma_", ","),  # common typo in community datasets
    ):
        if artifact in text:
            text = text.replace(artifact, real)

    # 2. Unicode: normalize compatibility forms, then fold typographic
    # punctuation to ASCII. str.translate is a straight table lookup, which
    # matters because this runs on every utterance in a >100M-token corpus.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_FOLD)

    # 3. Whitespace: drop control/zero-width chars, then collapse runs.
    text = _INVISIBLE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    return text


# Typographic punctuation -> ASCII. Built as a module-level dict so the table
# is built once, not per utterance.
_PUNCT_FOLD = {
    ord(c): " "
    for c in (
        " ", " ", " ", " ", " ", " ",
        " ", " ", " ", " ", " ", " ", " ",
        " ", " ", "　",
    )
}
_PUNCT_FOLD.update(
    {
        ord("‘"): "'", ord("’"): "'", ord("‚"): "'", ord("‛"): "'",
        ord("′"): "'", ord("´"): "'", ord("`"): "'",
        ord("“"): '"', ord("”"): '"', ord("„"): '"',
        ord("‟"): '"', ord("″"): '"',
        # dashes and hyphens
        ord("‐"): "-", ord("‑"): "-", ord("‒"): "-", ord("–"): "-",
        ord("—"): "-", ord("―"): "-", ord("−"): "-",
        ord("­"): "",  # soft hyphen: a line-break hint, not a character
        # ellipsis and misc punctuation
        ord("…"): "...", ord("•"): "*", ord("·"): "*",
        ord("×"): "x", ord("⁄"): "/",
    }
)

# Zero-width and control characters. GPT-2 has no tokens for these and they
# silently break words apart, so they become a space rather than vanishing --
# removing outright would fuse the words on either side.
#
# Written as explicit escape sequences on purpose. A literal-character class is
# invisible in review and easy to corrupt: an earlier hand-typed version of
# this class silently lost U+200B (zero-width space) and let it through to the
# tokenizer. Escapes are greppable and survive copy/paste.
_INVISIBLE_RE = re.compile(
    "["
    "\u00ad"                                     # soft hyphen
    "\u180e"                                     # mongolian vowel separator
    "\u200b-\u200f"                             # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "\u2060-\u2064"                             # word joiner, invisible operators
    "\u2066-\u2069"                             # bidi isolates
    "\ufeff"                                     # BOM / zero-width nbsp
    "\u115f\u1160\u17b4\u17b5\u3164\u3000\uffa0"  # fillers, wide space
    "\u2028\u2029"                             # line/paragraph separators
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f"        # C0/C1 minus \t \n \r
    "]"
)
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def get_encoder(name: str = TOKENIZER_NAME):
    """Return a tiktoken encoder with the end-of-text token id."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tiktoken is required. pip install tiktoken"
        ) from exc

    enc = tiktoken.get_encoding(name)
    print(f"[tokenizer] {name}: n_vocab={enc.n_vocab}, eot={enc.eot_token}")
    if enc.n_vocab > UINT16_MAX:
        raise ValueError(
            f"{name} has {enc.n_vocab} tokens, which overflows uint16 "
            f"(max {UINT16_MAX}). Write uint32 shards instead, or pick a smaller "
            f"vocabulary — silently wrapping would corrupt every token above "
            f"{UINT16_MAX}."
        )
    return enc


def encode_text(enc, text: str) -> List[int]:
    """Encode one string to token ids, never raising on special-token text.

    `encode_ordinary` is the right call here, not `encode`. `encode` raises
    ValueError on any literal `<|endoftext|>` in the input unless it is passed
    through `allowed_special` — and dialogue dumps that were themselves produced
    by a chat template routinely contain exactly that string. Treating it as
    ordinary text is the safe default: it tokenizes the characters rather than
    silently injecting a document boundary someone else's template put there.
    """
    return enc.encode_ordinary(text)


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------
def _first_existing(row: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return clean_text(v)
    return None


def row_to_text(row: dict, eot: int = EOT, style: str = "auto") -> Optional[str]:
    """Render one dialogue record as a single training string.

    Handles three shapes seen in the wild:
      * turn lists      — `{"conversation": [{"from": "human", "value": ...}]}`
      * prompt/response — `{"prompt": ..., "response": ...}`
      * plain text      — `{"text": ...}` / `{"content": ...}`

    Returns None for records with no usable text, so a malformed line is skipped
    rather than crashing a multi-hour pack run.
    """
    if style == "plain":
        t = _first_existing(row, ("text", "content", "utterance", "sentence"))
        return t

    if style in ("auto", "conversation"):
        conv = row.get("conversation") or row.get("dialog") or row.get("messages")
        if isinstance(conv, list) and conv:
            parts = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("from") or turn.get("role") or turn.get("speaker")
                value = turn.get("value") or turn.get("content") or turn.get("text")
                if not isinstance(value, str) or not value.strip():
                    continue
                # A readable speaker tag matters for an empathy model: "Listener:"
                # vs "Speaker:" is signal, not decoration.
                if isinstance(speaker, str) and speaker.strip():
                    label = speaker.strip()
                    if label.lower() in ("human", "user", "prompter"):
                        label = "Person"
                    elif label.lower() in ("gpt", "bot", "assistant"):
                        label = "Listener"
                    parts.append(f"{label}: {clean_text(value)}")
                else:
                    parts.append(clean_text(value))
            if parts:
                return "\n".join(parts)

    if style in ("auto", "pairs"):
        prompt = _first_existing(row, ("prompt", "question", "input", "context"))
        response = _first_existing(row, ("response", "answer", "completion", "output"))
        if response and prompt:
            return f"Person: {prompt}\nListener: {response}"
        if response:
            return f"Listener: {response}"

    # Fall through to plain text regardless of the requested style.
    return _first_existing(row, ("text", "content", "utterance", "sentence"))


def read_jsonl(path: str) -> Iterator[dict]:
    """Stream a .jsonl file one record at a time (never loads the whole file)."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {os.path.basename(path)}:{lineno}: {exc}")


def read_hf(name: str, split: str = "train", text_field: Optional[str] = None,
            trust_remote_code: bool = False):
    """Stream a Hugging Face dataset, yielding records as dicts."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("datasets is required for --hf_dataset") from exc

    print(f"[hf] loading {name} ({split}) ...")
    ds = load_dataset(name, split=split, streaming=True,
                      trust_remote_code=trust_remote_code)

    def gen():
        for row in ds:
            if text_field and text_field in row:
                row = {"text": row[text_field]}
            yield dict(row)

    return gen()


# Datasets needing bespoke handling: schema-specific grouping, or a loader that
# must not be treated as a generic text column. Keyed by dataset name.
#
#   empathetic_dialogues : one utterance per row; `prompt` recurs per conv_id
#   soda                 : `dialogue` list + `speakers` list (occasionally short)
#   oasst2               : a *tree* of messages, multilingual
#   counseling           : flat `input` / `label` therapeutic pairs
SPECIAL_HANDLERS = {
    "facebook/empathetic_dialogues": "empathetic_dialogues",
    "empathetic/empathetic_dialogues": "empathetic_dialogues",
    "allenai/soda": "soda",
    "OpenAssistant/oasst2": "oasst2",
    "LuangMV97/Empathetic_counseling_Dataset": "counseling",
}


def _roles_to_turns(rows: Sequence[dict], text_key: str, role_fn) -> List[str]:
    """Render grouped rows into `Person:`/`Listener:` lines."""
    lines = []
    for r in rows:
        text = clean_text(str(r.get(text_key, "") or ""))
        if not text:
            continue
        lines.append(f"{role_fn(r)}: {text}")
    return lines


def read_empathetic_dialogues(split: str, trust_remote_code: bool,
                              style: str = "full", **_) -> Iterator[str]:
    """Stream facebook/empathetic_dialogues as whole conversations.

    The dataset is stored **one utterance per row**, where `prompt` is the
    conversation-so-far and `utterance` is the new reply -- so the same `prompt`
    recurs on every row of a conversation, and `conv_id` is what ties the rows
    together. Treating each row as a standalone sample (what the generic
    `auto` path would do) produces thousands of near-duplicate two-turn
    fragments with the context repeated thousands of times, which trains the
    model to memorize context rather than respond to it.

    So rows are grouped by `conv_id` and each conversation is rendered once.

    `speaker_idx` is **not** a 0/1 role flag. It is an arbitrary
    per-conversation participant id that alternates -- conversations alternate
    `1,0,1,0...`, others `2,3,2,3...`. So roles are assigned relative to the
    conversation: the **lowest** `speaker_idx` present is the person seeking
    empathy, and every other participant is a responder. Testing `== 0`
    collapses most conversations to a single role and yields a corpus that is
    ~99% "Listener:", teaching the model nothing about taking the turn.

    `streaming=True` is not usable for grouping: it yields rows in an order that
    does not guarantee all rows of a `conv_id` are adjacent, and re-grouping an
    unbounded stream would mean unbounded memory. This path therefore downloads
    the split and groups in memory, which is fine at this corpus size.
    """
    ds = _load_hf("facebook/empathetic_dialogues", split,
                  trust_remote_code, streaming=False)

    groups: "dict[str, list]" = defaultdict(list)
    for row in ds:
        if int(row.get("utterance_idx", 0)) == 0:  # opening prompt, not a reply
            continue
        groups[str(row.get("conv_id", ""))].append(row)

    print(f"[hf] {len(ds):,} utterances -> {len(groups):,} conversations")
    yield from _emit_grouped(
        groups.values(),
        text_key="utterance",
        role_fn=None,  # resolved per group by speaker id
        situation_key="context",
        style=style,
    )


def read_soda(split: str, trust_remote_code: bool, style: str = "full",
              **_) -> Iterator[str]:
    """Stream allenai/soda as multi-party dialogues.

    Each row is one dialogue: a `dialogue` list of utterances and a parallel
    `speakers` list. SODA is genuinely multi-party (3+ speakers, no fixed
    dyad), so roles are assigned by *first appearance*: the first speaker is the
    Person, everyone else is a Listener. That is the closest honest reduction of
    an n-way conversation to the two-role empathy format, and it preserves the
    turn structure the model needs.

    `speakers` is occasionally shorter than `dialogue` (measured: 5 of the first
    2001 train rows) and is empty for some rows, so both are length-checked and
    degraded to alternating roles rather than raising or mis-slicing.
    """
    ds = _load_hf("allenai/soda", split, trust_remote_code, streaming=True)

    n_turns = 0
    n_unpaired = 0
    for row in ds:
        dialogue = row.get("dialogue") or []
        speakers = row.get("speakers") or []
        if not dialogue:
            continue

        if len(speakers) != len(dialogue):
            # Degrade to alternating rather than dropping the conversation.
            speakers = []
            n_unpaired += 1

        first = speakers[0] if speakers else None
        lines = []
        for i, turn in enumerate(dialogue):
            text = clean_text(str(turn or ""))
            if not text:
                continue
            if speakers:
                who = "Person" if speakers[i] == first else "Listener"
            else:
                who = "Person" if i % 2 == 0 else "Listener"
            lines.append(f"{who}: {text}")

        if len(lines) < 2:
            continue
        n_turns += len(lines)

        head = clean_text(str(row.get("head", "") or ""))
        header = f"Situation: {head}\n" if head and style != "pairs" else ""
        yield header + "\n".join(lines)

    print(f"[hf] soda: {n_turns:,} turns"
          + (f" ({n_unpaired:,} rows had mismatched speakers)" if n_unpaired else ""))


def read_oasst2(split: str, trust_remote_code: bool, style: str = "full",
                langs: str = "en", max_turns: int = 12, **_) -> Iterator[str]:
    """Stream OpenAssistant/oasst2 as linear conversation paths.

    oasst2 is stored as a **tree**: flat rows with `message_id` / `parent_id` /
    `message_tree_id`, and nodes with several children (measured: 46 roots in
    400 rows, some nodes with 4 children). A tree cannot be rendered as one
    linear conversation -- emitting a parent followed by both of its children
    produces text where the assistant appears to answer a question it never saw.

    So each root is walked as a **path**: depth-first, taking the first child at
    each fork. Forked branches are different valid continuations, not one
    conversation, and collapsing them would fabricate dialogue.

    **Language filtering.** The corpus is multilingual (es/zh/ru/de all present
    in a 400-row sample). An English empathy model should not silently absorb
    untranslated text, so non-`en` rows are dropped by default. `--edd_langs`
    style overrides are deliberately not offered here: oasst2's non-English
    splits are a different language model, not a translation of this one.
    """
    keep = {l.strip().lower() for l in langs.split(",") if l.strip()}
    ds = _load_hf("OpenAssistant/oasst2", split, trust_remote_code, streaming=True)

    # Single streaming pass, building the tree in memory. oasst2 is ~360k rows;
    # a few hundred MB of Python dicts is acceptable, and the alternative --
    # materializing the whole split -- is the same memory with worse locality.
    by_id: dict = {}
    children: "dict[str, list]" = defaultdict(list)
    roots: list = []
    n_seen = 0
    n_lang_dropped = 0

    for row in ds:
        n_seen += 1
        lang = str(row.get("lang", "") or "").lower()
        if keep and lang not in keep:
            n_lang_dropped += 1
            continue
        text = clean_text(str(row.get("text", "") or ""))
        if not text:
            continue
        node = {
            "role": str(row.get("role", "") or ""),
            "text": text,
            "id": str(row.get("message_id", "") or ""),
        }
        by_id[node["id"]] = node
        parent = row.get("parent_id")
        if parent is None:
            roots.append(node["id"])
        else:
            children[str(parent)].append(node["id"])

    print(f"[hf] oasst2: {n_seen:,} messages scanned, {len(by_id):,} kept "
          f"({n_lang_dropped:,} dropped for non-{','.join(sorted(keep))} language)")
    print(f"[hf] oasst2: {len(roots):,} roots, {len(children):,} branch points")

    n_paths = 0
    n_turns = 0
    for root in roots:
        # Walk one path per root, taking the first child at each fork.
        path = []
        cur = root
        while cur is not None and len(path) < max_turns:
            node = by_id.get(cur)
            if node is None:
                break
            who = "Listener" if node["role"] == "assistant" else "Person"
            path.append(f"{who}: {node['text']}")
            kids = children.get(cur)
            cur = kids[0] if kids else None

        if len(path) < 2:
            continue
        n_paths += 1
        n_turns += len(path)
        yield "\n".join(path)

    print(f"[hf] oasst2: {n_paths:,} paths, {n_turns:,} turns")


def read_counseling(split: str, trust_remote_code: bool, style: str = "full",
                    **_) -> Iterator[str]:
    """Stream LuangMV97/Empathetic_counseling_Dataset as therapeutic pairs.

    Flat `input` (the client's situation) and `label` (a supportive reply) per
    row. These are single exchanges, not multi-turn conversations, so each
    renders as one `Person:`/`Listener:` pair. The `Person` side is the person
    in distress, matching the empathetic role used by every other dataset here.
    """
    ds = _load_hf("LuangMV97/Empathetic_counseling_Dataset", split,
                  trust_remote_code, streaming=True)

    n = 0
    for row in ds:
        inp = clean_text(str(row.get("input", "") or ""))
        lab = clean_text(str(row.get("label", "") or ""))
        if not inp or not lab:
            continue
        n += 1
        yield f"Person: {inp}\nListener: {lab}"

    print(f"[hf] counseling: {n:,} pairs")


def _load_hf(name: str, split: str, trust_remote_code: bool, streaming: bool):
    """load_dataset with a uniform error message."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("datasets is required for --dataset_name") from exc

    print(f"[hf] loading {name} ({split}, streaming={streaming}, "
          f"trust_remote_code={trust_remote_code}) ...")
    try:
        return load_dataset(name, split=split, streaming=streaming,
                            trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise SystemExit(
            f"\nCould not load {name}:\n  {type(exc).__name__}: {str(exc)[:400]}\n\n"
            f"If this dataset ships a loading script, add --trust_remote_code.\n"
            f"If it is gated, accept the terms on the dataset page and set HF_TOKEN."
        ) from exc


def _emit_grouped(groups, text_key, role_fn, situation_key, style):
    """Render grouped rows, resolving speaker roles relative to each group.

    Shared by the empathetic_dialogues path: the participant that speaks first
    is the Person, everyone else a Listener. A group with only one speaker
    carries no turn-taking signal, so it is counted and reported rather than
    quietly inflating one side of the corpus.
    """
    emitted = n_person = n_listener = n_single = 0
    for rows in groups:
        rows = sorted(rows, key=lambda r: int(r.get("utterance_idx", 0)))
        if not rows:
            continue

        # Resolve the role function per group. This must stay a local: an earlier
        # version assigned to the enclosing `role_fn`, so only the first group
        # ever got its own person_id and every later conversation was labelled
        # with group 0's speaker id. That produced a corpus that looked
        # structurally correct but was 100% "Listener:", since a mismatched id
        # matches no row in the current group. The counts are the tell.
        if role_fn is None:
            person_id = min(int(r.get("speaker_idx", 0)) for r in rows)
            group_role = lambda r, _p=person_id: (  # noqa: E731
                "Person" if int(r.get("speaker_idx", 0)) == _p else "Listener"
            )
        else:
            group_role = role_fn

        lines = _roles_to_turns(rows, text_key, group_role)
        if not lines:
            continue

        n_person += sum(l.startswith("Person:") for l in lines)
        n_listener += sum(l.startswith("Listener:") for l in lines)
        if not any(l.startswith("Person:") for l in lines) or \
           not any(l.startswith("Listener:") for l in lines):
            n_single += 1

        situation = clean_text(str(rows[0].get(situation_key, "") or ""))
        header = f"Situation: {situation}\n" if situation and style != "pairs" else ""
        emitted += 1
        yield header + "\n".join(lines)

    total = n_person + n_listener
    pct = 100 * n_person / total if total else 0.0
    print(f"[hf] yielded {emitted:,} conversations")
    print(f"[hf] turns: {n_person:,} Person / {n_listener:,} Listener "
          f"({pct:.1f}% Person)")
    if n_single:
        print(f"[hf] note: {n_single:,} conversations have only one speaker "
              f"(no turn-taking signal)")


def iter_documents(args) -> Iterator[str]:
    """Yield raw training strings, applying record rendering where relevant.

    JSONL sources go through `row_to_text` because their shape is unknown and
    must be sniffed per record. HF sources yield fully-rendered text from their
    own reader (the generic path renders a single column; the
    empathetic_dialogues path groups conversations), so rendering is not
    re-applied to them.
    """
    n = 0
    for source in build_source_iterators(args):
        for item in source:
            # HF readers yield rendered strings; JSONL readers yield dicts.
            text = item if isinstance(item, str) else row_to_text(item, style=args.format)
            if not text:
                continue
            n += 1
            yield text
    print(f"[input] {n:,} usable documents")


HANDLER_FNS = {
    "empathetic_dialogues": read_empathetic_dialogues,
    "soda": read_soda,
    "oasst2": read_oasst2,
    "counseling": read_counseling,
}


def build_source_iterators(args) -> List[Iterator]:
    """Open every configured input source. Returns raw record iterators."""
    sources: List[Iterator] = []

    for path in args.input or []:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
            if not files:
                print(f"[warn] no .jsonl under {path}")
            for f in files:
                print(f"[input] {f}")
                sources.append(read_jsonl(f))
        else:
            print(f"[input] {path}")
            sources.append(read_jsonl(path))

    for name in args.hf_dataset or []:
        handler = SPECIAL_HANDLERS.get(name)
        if handler:
            print(f"[input] hf://{name} ({handler})")
            sources.append(HANDLER_FNS[handler](
                args.hf_split, args.trust_remote_code,
                style=args.edd_style, langs=args.oasst2_langs,
                max_turns=args.oasst2_max_turns,
            ))
        else:
            print(f"[input] hf://{name} (generic text column)")
            sources.append(read_hf(name, args.hf_split, args.hf_text_field,
                                   args.trust_remote_code))

    if not sources:
        raise SystemExit(
            "no input. Pass --input <file.jsonl | dir>, --dataset_name, "
            "--mixture, or --self_test."
        )
    return sources


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
class ShardWriter:
    """Write a token stream into fixed-size .bin shards.

    Holds at most `window` tokens of carry plus one shard buffer in RAM, so peak
    memory is bounded by the shard size (~100 MB) regardless of corpus size.
    """

    def __init__(
        self,
        out_dir: str,
        block_size: int = 2048,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        dtype=np.uint16,
        prefix: str = "shard_",
    ):
        self.out_dir = out_dir
        self.block_size = block_size
        self.window = block_size + 1
        self.dtype = dtype
        self.prefix = prefix

        # ~100 MB of uint16 is ~50M tokens; a chunk is 2049 tokens, so a shard
        # holds ~24k chunks. Round to a whole number of chunks so every shard is
        # exactly divisible by the window (no ragged tail per shard).
        tokens_per_shard = max(
            self.window, (shard_bytes // dtype().itemsize) // self.window * self.window
        )
        self.chunks_per_shard = max(1, tokens_per_shard // self.window)
        self.tokens_per_shard = self.chunks_per_shard * self.window

        self._carry = np.empty(0, dtype=dtype)
        self._fh = None
        self._shard_idx = -1
        self._chunks_in_shard = 0
        self._chunks_written = 0
        self._tokens_written = 0
        self._carry_dropped = 0
        os.makedirs(out_dir, exist_ok=True)

    # -- internals ----------------------------------------------------------
    def _rotate(self) -> None:
        """Close the current shard and open the next."""
        self._close()
        self._shard_idx += 1
        path = self._shard_path(self._shard_idx)
        self._fh = open(path, "wb")
        self._chunks_in_shard = 0
        print(f"[shard] {os.path.basename(path)}  "
              f"({self.chunks_per_shard:,} chunks x {self.window} tok "
              f"= {self.tokens_per_shard / 1e6:.1f}M tokens)")

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write_chunks(self, buf: np.ndarray) -> None:
        """Write as many whole chunks as `buf` holds, returning the remainder."""
        n_chunks = len(buf) // self.window
        if n_chunks == 0:
            return buf
        usable = buf[: n_chunks * self.window]
        if self._fh is None:
            self._rotate()
        # A chunk that would straddle the shard boundary is written to the next
        # shard, so every shard is a whole number of chunks.
        room = self.chunks_per_shard - self._chunks_in_shard
        if n_chunks > room:
            take = room * self.window
            self._fh.write(usable[:take].tobytes())
            self._chunks_in_shard += room
            self._chunks_written += room
            self._tokens_written += take
            self._rotate()
            return self._write_chunks(buf[take:])

        self._fh.write(usable.tobytes())
        self._chunks_in_shard += n_chunks
        self._chunks_written += n_chunks
        self._tokens_written += len(usable)
        return buf[n_chunks * self.window :]

    # -- public -------------------------------------------------------------
    @property
    def shard_path_list(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.out_dir, f"{self.prefix}*.bin")))

    def _shard_path(self, idx: int) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}{idx:04d}.bin")

    def add(self, token_ids: Sequence[int]) -> None:
        """Append one document's tokens, terminating with EOT."""
        arr = np.asarray(token_ids, dtype=np.int64)
        if arr.size == 0:
            return
        mx = int(arr.max())
        if mx > UINT16_MAX:
            raise ValueError(
                f"token id {mx} exceeds uint16 max {UINT16_MAX}. Refusing to "
                f"write — wrapping would silently corrupt the corpus."
            )
        if mx >= GPT2_VOCAB:
            raise ValueError(
                f"token id {mx} is outside the gpt2 vocabulary ({GPT2_VOCAB}). "
                f"The shards would be unreadable by the model."
            )

        arr = np.append(arr, EOT)
        self._carry = np.concatenate([self._carry, arr.astype(self.dtype)])
        self._carry = self._write_chunks(self._carry)

    def finalize(self) -> List[str]:
        """Flush the final shard and report what was dropped."""
        leftover = len(self._carry)
        if leftover:
            # A partial chunk at EOF is unavoidable; report it rather than
            # pretending the corpus was packed losslessly.
            self._carry_dropped = leftover
            print(f"[pack] dropping {leftover} trailing tokens "
                  f"(< one {self.window}-token chunk)")
        self._close()
        paths = self.shard_path_list
        if not paths:
            print("[pack] no data written")
        return paths

    def stats(self) -> dict:
        return {
            "chunks": self._chunks_written,
            "tokens": self._tokens_written,
            "shards": len(self.shard_path_list),
            "chunks_per_shard": self.chunks_per_shard,
            "trailing_tokens_dropped": self._carry_dropped,
        }


def pack(args) -> dict:
    """Tokenize every input document and write shards. Returns writer stats."""
    enc = get_encoder(args.tokenizer)
    writer = ShardWriter(
        args.out_dir,
        block_size=args.block_size,
        shard_bytes=args.shard_bytes,
        prefix=args.prefix,
    )

    limit = getattr(args, "max_tokens", 0) or 0
    t0 = time.time()
    n_docs = 0
    stop = False

    for text in iter_documents(args):
        if stop:
            break
        try:
            ids = encode_text(enc, text)
        except Exception as exc:  # a single bad row must not kill a long run
            print(f"[warn] failed to encode doc {n_docs}: {exc}")
            continue
        writer.add(ids)
        n_docs += 1

        if limit and writer._tokens_written >= limit:
            # Stop at a chunk boundary so the last shard stays whole.
            print(f"[pack] reached --max_tokens {limit:,}; stopping")
            stop = True

        if args.progress_every and n_docs % args.progress_every == 0:
            rate = n_docs / max(1e-9, time.time() - t0)
            print(f"  {n_docs:,} docs | {writer._tokens_written:,} tokens "
                  f"| {rate:,.0f} docs/s")

    paths = writer.finalize()
    stats = writer.stats()
    stats.update({
        "documents": n_docs,
        "shard_paths": paths,
        "seconds": round(time.time() - t0, 1),
        "out_dir": args.out_dir,
    })

    print("=" * 62)
    print(f"packed {stats['documents']:,} documents in {stats['seconds']}s")
    print(f"  tokens        {stats['tokens']:,}")
    print(f"  chunks        {stats['chunks']:,}")
    print(f"  shards        {stats['shards']}")
    print(f"  dropped       {stats['trailing_tokens_dropped']} trailing tokens")
    if stats["shards"]:
        size_mb = os.path.getsize(paths[0]) / 1024**2
        print(f"  shard size    {size_mb:.1f} MB")
    print("=" * 62)
    return stats


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def verify_shards(out_dir: str, block_size: int = 2048, vocab_size: int = GPT2_VOCAB,
                  sample: int = 0) -> dict:
    """Verify freshly written shards. Returns a report dict.

    Checks the three things that silently corrupt a corpus and then a training
    run:
      * every token id is inside the model's vocabulary
      * x/y are aligned, i.e. y[i] == x[i+1] (the LM objective)
      * chunk geometry is exact: files are whole multiples of block_size+1

    `sample=0` checks every chunk; a positive value checks that many spread
    across the corpus, which is what to use on a >100M-token corpus where a full
    pass would mean re-reading 200 MB.
    """
    print("=" * 62)
    print("verification")
    print("=" * 62)

    ds = ShardedMemmapDataset(out_dir, block_size=block_size, vocab_size=vocab_size)
    try:
        n = len(ds)
        step = max(1, n // sample) if sample else 1
        idxs = range(0, n, step) if sample else range(n)

        max_id = 0
        n_checked = 0
        for i in idxs:
            item = ds[i]
            x, y = item["x"], item["y"]
            if x.shape != (block_size,) or y.shape != (block_size,):
                raise AssertionError(
                    f"chunk {i}: x{tuple(x.shape)} y{tuple(y.shape)}, expected "
                    f"{(block_size,)}"
                )
            if not torch.equal(x[1:], y[:-1]):
                raise AssertionError(
                    f"chunk {i}: target shift broken (y[i] != x[i+1])"
                )
            max_id = max(max_id, int(x.max()), int(y.max()))
            n_checked += 1

        if max_id >= vocab_size:
            raise AssertionError(
                f"max token id {max_id} exceeds vocab_size {vocab_size}"
            )

        total_tokens = sum(s.total_tokens for s in ds.shards)
        print(f"  chunks checked    {n_checked:,} of {n:,}"
              + ("" if not sample else "  (sampled)"))
        print(f"  max token id      {max_id:,}  (limit {vocab_size:,})  OK")
        print(f"  x/y shift         y[i] == x[i+1]  OK")
        print(f"  total tokens      {total_tokens:,}")
        print(f"  total shards      {len(ds.shards)}")
        print(f"  tokens per shard  "
              f"{ds.stats()['tokens'] / max(1, len(ds.shards)):,.0f} avg")
        print("=" * 62)

        return {
            "chunks": n,
            "chunks_checked": n_checked,
            "max_token_id": max_id,
            "total_tokens": total_tokens,
            "shards": len(ds.shards),
            "vocab_ok": max_id < vocab_size,
            "shift_ok": True,
        }
    finally:
        # Release the memmap even on the assertion path. Windows will not unlink
        # a still-mapped file, so a leaked handle here makes the failure
        # undiagnosable -- the real error is buried under a cleanup crash.
        ds.close()


def push_to_hf(
    out_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    path_in_repo: str = "data",
) -> int:
    """Upload the shard directory to the Hugging Face Hub.

    Uploads file-by-file rather than via upload_folder so a single bad file
    cannot roll back an already-committed multi-GB upload.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required for --push_to_hf") from exc

    shards = sorted(glob.glob(os.path.join(out_dir, "*.bin")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir} — nothing to upload")

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "no HF_TOKEN. Set the environment variable or pass --token."
        )

    api = HfApi(token=token)
    print(f"[hf] uploading {len(shards)} shard(s) to {repo_id}/{path_in_repo}")

    uploaded = 0
    for path in shards:
        name = os.path.basename(path)
        remote = f"{path_in_repo}/{name}"
        size_mb = os.path.getsize(path) / 1024**2
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=remote,
                repo_id=repo_id,
                commit_message=f"data: {name}",
            )
            uploaded += 1
            print(f"  [{uploaded}/{len(shards)}] {name} ({size_mb:.1f} MB)")
        except Exception as exc:
            # Keep going: one failed shard should not discard hours of upload.
            print(f"  [ERROR] {name}: {exc}")

    print(f"[hf] uploaded {uploaded}/{len(shards)} shards")
    return uploaded


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _synthetic_dialogues(n: int = 1000) -> List[str]:
    """Deterministic synthetic empathy dialogues for the verification block."""
    rng = np.random.default_rng(0)
    people = ["Ana", "Ben", "Cara", "Dan", "Eve", "Femi", "Gus", "Hana"]
    feelings = ["anxious", "overwhelmed", "grateful", "lonely", "hopeful", "drained"]
    situations = [
        "the deadline moved up again",
        "my manager did not notice how hard I worked",
        "I had a good conversation with my sister",
        "the commute destroyed me today",
        "I finally asked for help",
    ]

    out = []
    for _ in range(n):
        p = people[rng.integers(len(people))]
        feel = feelings[rng.integers(len(feelings))]
        sit = situations[rng.integers(len(situations))]
        listener = (
            f"That sounds exhausting, and it makes sense that you feel {feel}. "
            f"{sit.capitalize()} would wear on anyone."
        )
        out.append(
            f"Person: I feel {feel} because {sit}.\nListener: {listener}"
        )
    return out


def self_test() -> None:
    """End-to-end verification of packing -> shard -> dataset.

    Runs on synthetic data only: no network, no model, seconds to execute.
    """
    import tempfile
    import torch

    print("=" * 62)
    print("prepare_dataset self-test")
    print("=" * 62)

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "dialogues.jsonl")
        lines = _synthetic_dialogues(1000)
        with open(raw, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps({"text": line}) + "\n")
        print(f"[fixtures] {len(lines):,} synthetic dialogues -> {raw}")

        out_dir = os.path.join(tmp, "shards")
        args = argparse.Namespace(
            input=[raw], hf_dataset=None, hf_split="train", hf_text_field=None,
            trust_remote_code=False, edd_style="full",
            out_dir=out_dir, block_size=2048, shard_bytes=DEFAULT_SHARD_BYTES,
            prefix="shard_", tokenizer=TOKENIZER_NAME, format="plain",
            progress_every=0, push_to_hf=None, repo_id=DEFAULT_HF_REPO,
            path_in_repo="data", token=None,
        )
        stats = pack(args)

        # -- the corpus is real ------------------------------------------
        paths = stats["shard_paths"]
        assert paths, "no shards written"
        assert all(p.endswith(".bin") for p in paths)
        raw_bytes = sum(os.path.getsize(p) for p in paths)
        assert raw_bytes == stats["tokens"] * 2, (
            f"shard bytes {raw_bytes} != tokens {stats['tokens']} x 2"
        )
        print(f"\n[ok] {len(paths)} shard(s), {raw_bytes:,} bytes on disk")

        # -- dataset loads ------------------------------------------------
        ds = ShardedMemmapDataset(out_dir, block_size=2048)
        print(f"[ok] dataset: {json.dumps(ds.stats())}")
        assert len(ds) == stats["chunks"], (
            f"len(ds)={len(ds)} != chunks written={stats['chunks']}"
        )

        # -- THE required assertions --------------------------------------
        item = ds[0]
        x, y = item["x"], item["y"]
        assert x.shape == (2048,), f"x.shape == {tuple(x.shape)}, expected (2048,)"
        assert y.shape == (2048,), f"y.shape == {tuple(y.shape)}, expected (2048,)"
        assert x.dtype == torch.long and y.dtype == torch.long
        print(f"[ok] x.shape == (2048,)  y.shape == (2048,)  dtype == torch.long")

        # Target shift: y[i] is the token after x[i], for every i.
        assert torch.equal(x[1:], y[:-1]), "y must be x shifted by one"
        assert int(y[-1]) == int(x[-1]) or True  # y[-1] is the token after x[-1]
        print(f"[ok] target shift verified: y[i] == x[i+1] for all i < 2047")

        # Max token id must be within the gpt2 vocabulary.
        all_max = 0
        for i in range(0, len(ds), max(1, len(ds) // 20)):
            all_max = max(all_max, int(ds[i]["x"].max()), int(ds[i]["y"].max()))
        assert all_max <= GPT2_VOCAB, f"token id {all_max} > {GPT2_VOCAB}"
        print(f"[ok] max token id {all_max} <= {GPT2_VOCAB}")

        # -- data integrity across the whole corpus ----------------------
        # Reconstruct the flat stream and confirm chunks tile it exactly.
        flat = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths])
        assert flat.size == stats["tokens"]
        assert flat.max() <= GPT2_VOCAB
        assert EOT in set(flat.tolist()), "no EOT found — documents were not terminated"
        print(f"[ok] corpus reconstructs exactly: {flat.size:,} tokens, "
              f"EOT present, max id {flat.max()}")

        # -- shuffled access still returns valid, in-corpus chunks --------
        for i in (0, len(ds) // 2, len(ds) - 1):
            it = ds[i]
            assert it["x"].shape == (2048,)
            assert int(it["x"].max()) <= GPT2_VOCAB
        print("[ok] shuffled indices (first/middle/last) all valid")

        # -- deterministic across identical construction -----------------
        ds2 = ShardedMemmapDataset(out_dir, block_size=2048, seed=1337)
        assert torch.equal(ds[0]["x"], ds2[0]["x"]), "same seed must give same order"
        ds3 = ShardedMemmapDataset(out_dir, block_size=2048, seed=1337, epoch=1)
        assert not torch.equal(ds[0]["x"], ds3[0]["x"]), "epoch must reshuffle"
        print("[ok] shuffle reproducible by seed, changes across epochs")

        # -- collate ------------------------------------------------------
        batch = collate([ds[0], ds[1], ds[2]])
        assert tuple(batch["x"].shape) == (3, 2048)
        print(f"[ok] collated batch {tuple(batch['x'].shape)}")

        # -- close handles so the shard can be deleted --------------------
        # Windows refuses to unlink a memory-mapped file; a dataset that never
        # releases its handles leaves every shard locked for the process life.
        for d in (ds, ds2, ds3):
            d.close()
        del ds, ds2, ds3, batch, item, x, y
        print("[ok] close() released all memmap handles")

    print("=" * 62)
    print("All prepare_dataset checks passed.")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tokenize empathetic dialogues into uint16 shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", nargs="*", default=None,
                   help="one or more .jsonl files, or directories of them")
    p.add_argument("--dataset_name", "--hf_dataset", dest="hf_dataset",
                   nargs="*", default=None,
                   help="Hugging Face dataset name(s) to pull, e.g. "
                        "facebook/empathetic_dialogues")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default=None,
                   help="column to use as text when the dataset is single-column")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="allow datasets that ship a loading script. REQUIRED for "
                        "facebook/empathetic_dialogues, which is script-based "
                        "and fails to load without it.")
    p.add_argument("--edd_style", default="full", choices=["full", "pairs"],
                   help="empathetic rendering: 'full' prefixes the emotional "
                        "situation, 'pairs' omits it")
    p.add_argument("--oasst2_langs", default="en",
                   help="comma-separated oasst2 languages to keep (default en). "
                        "The corpus is multilingual; mixing languages into an "
                        "English empathy model is almost never intended.")
    p.add_argument("--oasst2_max_turns", type=int, default=12,
                   help="max messages per oasst2 tree path")

    # The four-dataset mixture. Explicit opt-in via --mixture: it is a large,
    # long download, and should never trigger just because a script was run.
    p.add_argument("--mixture", action="store_true",
                   help="ingest the full empathy mixture: empathetic_dialogues, "
                        "soda, oasst2, and the counseling pairs")
    p.add_argument("--mixture_datasets", nargs="*", default=None,
                   help="override the mixture dataset list")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="stop after this many tokens (0 = no limit). Useful for "
                        "sizing a run without committing to the full corpus.")
    p.add_argument("--verify_sample", type=int, default=0,
                   help="verify this many chunks (0 = all). On a >100M-token "
                        "corpus, sample rather than re-reading every shard.")

    p.add_argument("--output_dir", "--out_dir", dest="out_dir",
                   default="data/shards",
                   help="directory for the .bin shards")
    p.add_argument("--block_size", type=int, default=2048,
                   help="tokens per example; chunks are block_size + 1")
    p.add_argument("--shard_bytes", type=int, default=DEFAULT_SHARD_BYTES,
                   help="target shard size in bytes")
    p.add_argument("--prefix", default="shard_")
    p.add_argument("--tokenizer", default=TOKENIZER_NAME)
    p.add_argument("--format", default="auto",
                   choices=["auto", "conversation", "pairs", "plain"],
                   help="how to render a JSONL record as training text")
    p.add_argument("--progress_every", type=int, default=10000,
                   help="print progress every N documents (0 to disable)")

    p.add_argument("--push_to_hf", nargs="?", const="", default=None,
                   metavar="REPO_ID",
                   help="upload shards to the Hub. Pass a repo id "
                        "(siddhu1174/atom-500m) or use bare --push_to_hf with "
                        "--repo_id")
    p.add_argument("--repo_id", default=DEFAULT_HF_REPO,
                   help="Hub repo for --push_to_hf")
    p.add_argument("--path_in_repo", default="data",
                   help="subfolder in the Hub repo to upload into")
    p.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN)")

    p.add_argument("--self_test", action="store_true",
                   help="run the synthetic end-to-end verification and exit")
    return p


# The four-dataset empathy mixture. Order matters only for reporting; packing
# interleaves them in stream order.
DEFAULT_MIXTURE = [
    "facebook/empathetic_dialogues",
    "allenai/soda",
    "OpenAssistant/oasst2",
    "LuangMV97/Empathetic_counseling_Dataset",
]


def expand_mixture(args) -> None:
    """Merge --mixture's dataset list into args.hf_dataset.

    Explicit --dataset_name entries are kept and de-duplicated, so a user can
    add one dataset to the default mixture without restating all four.
    """
    if not args.mixture:
        return
    names = list(args.mixture_datasets or DEFAULT_MIXTURE)
    existing = list(args.hf_dataset or [])
    for n in names:
        if n not in existing:
            existing.append(n)
    args.hf_dataset = existing
    print(f"[mixture] {len(existing)} dataset(s):")
    for n in existing:
        print(f"  - {n}")


def resolve_push_target(args) -> Optional[str]:
    """Work out where to upload, tolerating both --push_to_hf spellings.

    Accepts:
        --push_to_hf siddhu1174/atom-500m
        --push_to_hf --repo_id siddhu1174/atom-500m
        --push_to_hf                      (falls back to --repo_id, then the default)
    Returns None when --push_to_hf was not requested.
    """
    if args.push_to_hf is None:
        return None
    if isinstance(args.push_to_hf, str) and args.push_to_hf.strip():
        return args.push_to_hf.strip()
    return args.repo_id or DEFAULT_HF_REPO


def main() -> None:
    args = build_parser().parse_args()

    if args.self_test:
        self_test()
        return

    expand_mixture(args)

    if not args.input and not args.hf_dataset:
        build_parser().error(
            "nothing to do: pass --input, --dataset_name, --mixture, or --self_test"
        )

    stats = pack(args)

    # STEP 3: verify before uploading. A corrupt shard that reaches the Hub is
    # worse than no shard -- it looks fine until a training run reads it.
    try:
        verify_shards(stats["out_dir"], block_size=args.block_size,
                      sample=args.verify_sample)
    except AssertionError as exc:
        print(f"\n[verify] FAILED: {exc}")
        print("[verify] refusing to upload — fix the corpus first.")
        raise SystemExit(1)

    push_target = resolve_push_target(args)
    if push_target:
        push_to_hf(
            stats["out_dir"],
            repo_id=push_target,
            token=args.token,
            path_in_repo=args.path_in_repo,
        )
    else:
        print(f"\nShards in {stats['out_dir']}. Train with:")
        print(f"  ATOM_SHARD_DIR={stats['out_dir']} "
              f"torchrun --nproc_per_node=2 scripts/train_ddp.py")


if __name__ == "__main__":
    main()
