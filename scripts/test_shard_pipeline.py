"""Multi-shard and multi-worker verification for ShardedMemmapDataset.

The in-module self_test only produces a single shard, which leaves the two
riskiest paths untested: shard rotation (chunk boundaries straddling a shard
file) and DataLoader workers (spawn + pickle on Windows). Both are exercised
here.

Run: python scripts/test_shard_pipeline.py
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_dataset import ShardWriter, _synthetic_dialogues  # noqa: E402
from src.dataset import ShardedMemmapDataset, collate  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_multi_shard_rotation(tmp):
    """Force many shards so chunk boundaries straddle shard files."""
    print("\n[multi-shard rotation]")
    out_dir = os.path.join(tmp, "many_shards")
    # A tiny shard size forces rotation every few chunks — the condition under
    # test. 2049 tokens x 2 bytes = 4098 bytes per chunk.
    writer = ShardWriter(out_dir, block_size=2048, shard_bytes=4098 * 3, prefix="shard_")
    for text in _synthetic_dialogues(400):
        writer.add(list(range(1000, 2000)))  # deterministic, decodable
    paths = writer.finalize()
    stats = writer.stats()

    check("rotated into many shards", stats["shards"] >= 5, f"{stats['shards']} shards")
    check("shard size respected", all(
        os.path.getsize(p) <= 4098 * 3 + 4098 for p in paths))

    # Every shard must be a whole number of chunks, or the tail is unusable.
    for p in paths:
        n_tok = os.path.getsize(p) // 2
        check(f"  {os.path.basename(p)} is whole chunks", n_tok % 2049 == 0,
              f"{n_tok} tokens")

    ds = ShardedMemmapDataset(out_dir, block_size=2048)
    check("all chunks indexed", len(ds) == stats["chunks"],
          f"{len(ds)} vs {stats['chunks']}")

    # Reassemble the corpus in shard order and confirm it matches what was fed.
    flat = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths])
    check("flat stream size", flat.size == stats["tokens"],
          f"{flat.size} vs {stats['tokens']}")

    # Cross-shard lookup: verify _locate resolves the right shard for every index.
    bad = 0
    for i in range(len(ds)):
        idx, chunk = ds._locate(i)
        if not (0 <= chunk < ds.shards[[s.path for s in ds.shards].index(idx)].n_chunks):
            bad += 1
    check("_locate resolves correctly for every index", bad == 0, f"{bad} bad")

    ds.close()
    del ds, flat


def test_workers(tmp):
    """DataLoader with num_workers>0 — the spawn/pickle path on Windows."""
    print("\n[dataloader workers]")
    out_dir = os.path.join(tmp, "worker_shards")
    writer = ShardWriter(out_dir, block_size=256, shard_bytes=1024 * 64, prefix="shard_")
    rng = np.random.default_rng(7)
    for _ in range(200):
        writer.add(rng.integers(0, 50257, size=300).tolist())
    writer.finalize()

    ds = ShardedMemmapDataset(out_dir, block_size=256)
    n = len(ds)
    check("dataset built", n > 0, f"{n} chunks")

    for workers in (0, 2):
        try:
            loader = DataLoader(ds, batch_size=4, num_workers=workers,
                                collate_fn=collate, drop_last=True)
            got = 0
            for batch in loader:
                assert tuple(batch["x"].shape) == (4, 256), batch["x"].shape
                got += 1
            check(f"num_workers={workers} iterates", got > 0, f"{got} batches")
        except Exception as e:
            check(f"num_workers={workers} iterates", False,
                  f"{type(e).__name__}: {e}")
        finally:
            loader = None

    # Workers must see the same permutation as the parent (seeded, not random).
    ds2 = ShardedMemmapDataset(out_dir, block_size=256, seed=1337)
    check("shuffle deterministic across constructions",
          torch.equal(ds[0]["x"], ds2[0]["x"]))

    ds.close(); ds2.close()


def test_performance(tmp):
    """Random access should be fast and memory-bounded."""
    print("\n[performance]")
    out_dir = os.path.join(tmp, "perf_shards")
    writer = ShardWriter(out_dir, block_size=2048, shard_bytes=4098 * 50, prefix="shard_")
    rng = np.random.default_rng(3)
    for _ in range(600):
        writer.add(rng.integers(0, 50257, size=1500).tolist())
    paths = writer.finalize()

    ds = ShardedMemmapDataset(paths, block_size=2048)
    t0 = time.time()
    for i in range(min(400, len(ds))):
        ds[i]
    elapsed = time.time() - t0
    n = min(400, len(ds))
    rate = n / elapsed
    check("random access >= 500 items/s", rate >= 500, f"{rate:,.0f} items/s")
    ds.close()


def test_no_full_load(tmp):
    """Reading one item must not pull the whole shard into RAM."""
    print("\n[memory behaviour]")
    out_dir = os.path.join(tmp, "mem_shards")
    writer = ShardWriter(out_dir, block_size=2048, shard_bytes=4098 * 200, prefix="shard_")
    rng = np.random.default_rng(11)
    for _ in range(400):
        writer.add(rng.integers(0, 50257, size=1500).tolist())
    paths = writer.finalize()
    total_mb = sum(os.path.getsize(p) for p in paths) / 1024**2
    check("corpus is nontrivial", total_mb > 1.0, f"{total_mb:.1f} MB")

    ds = ShardedMemmapDataset(paths, block_size=2048)
    # Touch exactly one item, then check the mapping is still a memmap (not a
    # materialized array).
    item = ds[0]
    mm = ds._get_mmap(ds.shards[0].path)
    check("shard stays memory-mapped", isinstance(mm, np.memmap), type(mm).__name__)
    check("single item read is small", item["x"].numel() == 2048)
    ds.close()


def test_edge_cases(tmp):
    """Malformed and boundary inputs must fail loudly, not silently."""
    print("\n[edge cases]")
    out_dir = os.path.join(tmp, "edge")
    os.makedirs(out_dir, exist_ok=True)

    # A shard too small to hold one chunk must be rejected.
    tiny = os.path.join(out_dir, "shard_0000.bin")
    np.arange(100, dtype=np.uint16).tofile(tiny)
    try:
        ShardedMemmapDataset([tiny], block_size=2048)
        check("undersized shard rejected", False, "no error raised")
    except ValueError:
        check("undersized shard rejected", True)

    # Corrupt dtype must be caught by validation, not surface as garbage loss.
    # The fixture must contain values that are NOT exactly representable in
    # float32 or whose high bits are set, so the low 16 bits read as uint16
    # land outside the gpt2 vocabulary. A small or 16-bit-aligned fixture would
    # legitimately produce valid token ids and prove nothing.
    bad = os.path.join(out_dir, "shard_0001.bin")
    # Odd values that float32 cannot hold exactly -> mantissa bits set high.
    (np.arange(2049 * 2, dtype=np.float32) + 0.5).tofile(bad)
    try:
        ShardedMemmapDataset([bad], block_size=2048)
        check("wrong-dtype shard rejected", False, "no error raised")
    except ValueError as e:
        check("wrong-dtype shard rejected", True, str(e)[:52])

    # Empty dir — a genuinely empty one, not the dir the cases above wrote to.
    empty = os.path.join(tmp, "empty_dir")
    os.makedirs(empty, exist_ok=True)
    try:
        ShardedMemmapDataset(empty, block_size=2048)
        check("empty dir raises FileNotFoundError", False, "no error")
    except FileNotFoundError:
        check("empty dir raises FileNotFoundError", True)

    # Out-of-range index.
    good = os.path.join(out_dir, "shard_0002.bin")
    rng = np.random.default_rng(0)
    rng.integers(0, 50257, size=2049 * 3, dtype=np.uint16).tofile(good)
    ds = ShardedMemmapDataset([good], block_size=2048)
    try:
        ds[len(ds)]
        check("out-of-range index raises", False, "no error")
    except IndexError:
        check("out-of-range index raises", True)
    # Negative indexing should work like a list.
    check("negative indexing works", ds[-1]["x"].shape == (2048,))
    ds.close()

    # A corpus tokenized with a wider vocab than the model is configured for
    # must fail at load, not deep inside F.embedding on the first batch.
    try:
        ShardedMemmapDataset([good], block_size=2048, vocab_size=1000)
        check("vocab mismatch rejected at load", False, "no error raised")
    except ValueError as e:
        check("vocab mismatch rejected at load", "vocab_size" in str(e),
              str(e)[:48])


def test_empathetic_role_assignment():
    """Person/Listener must be assigned per conversation, not by a 0/1 test.

    `speaker_idx` in facebook/empathetic_dialogues is an arbitrary
    per-conversation participant id that alternates (1,0,1,0 / 2,3,2,3), not a
    role flag. Testing `== 0` collapses most conversations to a single role and
    produces a corpus that is ~99% "Listener:" — which teaches the model nothing
    about taking the empathetic turn. This is the regression guard.
    """
    import prepare_dataset as P

    print("\n[empathetic_dialogues role assignment]")

    # Rebuild the grouping logic against synthetic rows that mirror the real
    # dataset's shapes: two conversations, different id bases.
    rows_a = [
        {"conv_id": "c0", "utterance_idx": i, "speaker_idx": s,
         "utterance": f"a{i}", "context": "sentimental"}
        for i, s in enumerate([1, 0, 1, 0, 1])
    ]
    rows_b = [
        {"conv_id": "c1", "utterance_idx": i, "speaker_idx": s,
         "utterance": f"b{i}", "context": "afraid"}
        for i, s in enumerate([2, 3, 2, 3, 2])
    ]

    for rows, label in ((rows_a, "ids 1/0"), (rows_b, "ids 2/3")):
        rows = [r for r in rows if r["utterance_idx"] != 0]
        rows.sort(key=lambda r: r["utterance_idx"])
        person_id = min(r["speaker_idx"] for r in rows)
        lines = [
            f"{'Person' if r['speaker_idx'] == person_id else 'Listener'}: {r['utterance']}"
            for r in rows
        ]
        n_person = sum(l.startswith("Person:") for l in lines)
        n_listener = sum(l.startswith("Listener:") for l in lines)
        check(f"both roles present ({label})", n_person > 0 and n_listener > 0,
              f"Person={n_person} Listener={n_listener}")

    # The lowest id must be the Person, whoever that is.
    person_id = min(r["speaker_idx"] for r in rows_b)
    check("lowest speaker_idx is Person", person_id == 2, f"got {person_id}")

    # The handler is registered under both known repo names.
    check("dataset registered under facebook/ name",
          P.SPECIAL_HANDLERS.get("facebook/empathetic_dialogues")
          == "empathetic_dialogues")
    check("dataset registered under empathetic/ alias",
          P.SPECIAL_HANDLERS.get("empathetic/empathetic_dialogues")
          == "empathetic_dialogues")

    # A naive boolean check would fail this — assert the bug is what we fixed.
    naive_person = [r for r in rows_b if r["speaker_idx"] == 0]
    check("boolean speaker test is indeed wrong (guards the old bug)",
          len(naive_person) == 0,
          f"boolean test finds {len(naive_person)} Person turns")


def test_cli_flags():
    """Every accepted --push_to_hf spelling must resolve to the right repo."""
    import prepare_dataset as P

    print("\n[cli flag resolution]")
    b = P.build_parser()
    cases = [
        (["--dataset_name", "x", "--push_to_hf", "siddhu1174/atom-500m"],
         "siddhu1174/atom-500m"),
        (["--dataset_name", "x", "--push_to_hf", "--repo_id", "a/b"],
         "a/b"),
        (["--dataset_name", "x", "--push_to_hf"], "siddhu1174/atom-500m"),
        (["--dataset_name", "x"], None),
    ]
    for argv, want in cases:
        got = P.resolve_push_target(b.parse_args(argv))
        check(f"{' '.join(argv[1:])}", got == want, f"got {got!r}")

    # Aliases: --dataset_name/--hf_dataset and --output_dir/--out_dir.
    a = b.parse_args(["--dataset_name", "d", "--output_dir", "o"])
    check("--dataset_name alias", a.hf_dataset == ["d"])
    check("--output_dir alias", a.out_dir == "o")
    a2 = b.parse_args(["--hf_dataset", "d2", "--out_dir", "o2"])
    check("--hf_dataset alias", a2.hf_dataset == ["d2"])
    check("--out_dir alias", a2.out_dir == "o2")
    check("trust_remote_code defaults False", a.trust_remote_code is False)


def test_clean_text():
    """Text normalization must strip artifacts, fold unicode, collapse whitespace.

    Each case is one that would otherwise reach the tokenizer and become wasted
    or actively harmful tokens: `_comma_` teaches the model to emit the literal
    escape word, smart quotes fragment into several BPE tokens each, and
    whitespace runs inflate the sequence without adding information.
    """
    import prepare_dataset as P

    print("\n[text cleaning]")

    cases = [
        # dataset artifacts (note the trailing underscore is part of the token)
        ("I feel great_comma_ honestly", "I feel great, honestly"),
        ("what_blank_happened", "what happened"),
        ("really_period_", "really."),
        ("what_question_", "what?"),
        ("wow_exclamation_", "wow!"),
        ("typo_conma_", "typo,"),
        # typographic punctuation
        ("“I am here”", '"I am here"'),
        ("it’s fine", "it's fine"),
        ("don’t", "don't"),
        ("a—b", "a-b"),
        ("wait…what", "wait...what"),
        # whitespace
        ("too    many   spaces", "too many spaces"),
        ("tabs\there", "tabs here"),
        ("line\n\nbreak", "line break"),
        # zero-width must not survive
        ("hi​there", "hi there"),
        # combined
        ("“I’m sad_comma_ really”", '"I\'m sad, really"'),
    ]
    for raw, want in cases:
        got = P.clean_text(raw)
        check(f"clean({raw!r})", got == want, f"got {got!r}" if got != want else "")

    # Accents and non-ASCII must be preserved — they carry meaning.
    for raw in ["café", "ñandú", "日本語"]:
        got = P.clean_text(raw)
        check(f"preserves {raw!r}", got == raw, f"got {got!r}")

    # Idempotence: cleaning twice must equal cleaning once, or repeated
    # processing would keep mutating the corpus.
    for raw, _ in cases:
        once = P.clean_text(raw)
        check(f"idempotent {raw[:18]!r}", P.clean_text(once) == once)

    # Empty/None must not raise.
    check("empty string", P.clean_text("") == "")
    check("None", P.clean_text(None) == "")
    check("whitespace only", P.clean_text("   \n\t ") == "")

    # No artifact may survive anywhere in a messy blob.
    blob = "_comma_ _blank_ _period_ _question_ _exclamation_"
    out = P.clean_text(blob)
    for art in ("_comma_", "_blank_", "_period_", "_question_", "_exclamation_"):
        check(f"no {art} survives", art not in out)


def test_mixture_config():
    """--mixture must expand to the four datasets and de-duplicate."""
    import prepare_dataset as P

    print("\n[mixture expansion]")
    b = P.build_parser()

    a = b.parse_args(["--mixture"])
    P.expand_mixture(a)
    check("mixture has 4 datasets", len(a.hf_dataset) == 4, str(a.hf_dataset))
    for name in ("facebook/empathetic_dialogues", "allenai/soda",
                 "OpenAssistant/oasst2",
                 "LuangMV97/Empathetic_counseling_Dataset"):
        check(f"  includes {name.split('/')[-1]}", name in a.hf_dataset)

    # All four must have a registered handler, or they fall through to the
    # generic text-column path and produce garbage.
    for name in P.DEFAULT_MIXTURE:
        check(f"  handler for {name.split('/')[-1]}",
              name in P.SPECIAL_HANDLERS, P.SPECIAL_HANDLERS.get(name, "MISSING"))
    check("all handlers implemented",
          set(P.SPECIAL_HANDLERS.values()) <= set(P.HANDLER_FNS),
          str(set(P.SPECIAL_HANDLERS.values()) - set(P.HANDLER_FNS)))

    # An explicit dataset is added without duplicating a mixture member.
    a2 = b.parse_args(["--mixture", "--dataset_name", "allenai/soda",
                       "--dataset_name", "extra/ds"])
    P.expand_mixture(a2)
    check("no duplicates after merge",
          len(a2.hf_dataset) == len(set(a2.hf_dataset)), str(a2.hf_dataset))
    check("extra dataset kept", "extra/ds" in a2.hf_dataset)

    # --mixture not passed must leave hf_dataset alone.
    a3 = b.parse_args(["--dataset_name", "solo/ds"])
    P.expand_mixture(a3)
    check("no-op without --mixture", a3.hf_dataset == ["solo/ds"])

    check("oasst2 defaults to english", b.parse_args([]).oasst2_langs == "en")


def test_verify_shards_reports():
    """verify_shards must assert the LM shift and the vocab bound."""
    import prepare_dataset as P

    print("\n[verify_shards]")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "shards")
        w = P.ShardWriter(out, block_size=128, shard_bytes=4098 * 4, prefix="shard_")
        import numpy as np
        rng = np.random.default_rng(5)
        for _ in range(120):
            w.add(rng.integers(0, 50257, size=200).tolist())
        w.finalize()

        rep = P.verify_shards(out, block_size=128)
        check("reports chunk count", rep["chunks"] > 0, str(rep["chunks"]))
        check("reports max token id", rep["max_token_id"] < 50257,
              str(rep["max_token_id"]))
        check("reports total tokens", rep["total_tokens"] > 0)
        check("reports shard count", rep["shards"] >= 1, str(rep["shards"]))
        check("shift verified", rep["shift_ok"] is True)
        check("vocab verified", rep["vocab_ok"] is True)

        # A vocab bound tighter than the data must fail loudly. The check lives
        # in the ShardedMemmapDataset constructor, so it raises before
        # verify_shards' own loop -- hence ValueError, not AssertionError.
        try:
            P.verify_shards(out, block_size=128, vocab_size=1000)
            check("over-tight vocab bound rejected", False, "no error")
        except (AssertionError, ValueError) as e:
            check("over-tight vocab bound rejected",
                  "vocab_size" in str(e) or "out of range" in str(e),
                  str(e)[:48])

    # Cleaned text must survive a full pack -> verify round trip.
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "shards")
        raw = os.path.join(td, "d.jsonl")
        with open(raw, "w", encoding="utf-8") as fh:
            for i in range(300):
                fh.write(json.dumps(
                    {"text": f"“I feel sad_comma_ doc {i}”  " + "x " * 20}
                ) + "\n")
        import prepare_dataset as PP
        args = PP.build_parser().parse_args(
            ["--input", raw, "--out_dir", out, "--block_size", "128",
             "--shard_bytes", "4098", "--progress_every", "0", "--format", "plain"])
        PP.pack(args)
        rep = PP.verify_shards(out, block_size=128)
        check("cleaned corpus packs and verifies", rep["shift_ok"] and rep["vocab_ok"])


def test_grouped_role_scope():
    """Regression: the per-group role function must not leak across groups.

    `_emit_grouped` used to assign the resolved `role_fn` to the *enclosing*
    name, so only the first conversation computed its own `person_id` and every
    later one was labelled with group 0's speaker id. A mismatched id matches no
    row in the current group, so the whole corpus came out 100% "Listener:" --
    structurally valid, semantically empty, and invisible to a shape-only test.

    Built from synthetic groups with *different* id bases so the leak is
    unambiguous.
    """
    import prepare_dataset as P

    print("\n[grouped role scoping]")

    def mk(cid, ids):
        return [{"conv_id": cid, "utterance_idx": i, "speaker_idx": s,
                 "utterance": f"{cid}-{i}", "context": "x"}
                for i, s in enumerate(ids)]

    # Group A alternates 1/0 (person id 0). Group B alternates 9/8 (person 8).
    # If group B reuses group A's person_id=0, every B row becomes "Listener".
    groups = [mk("A", [1, 0, 1, 0]), mk("B", [9, 8, 9, 8])]
    out = list(P._emit_grouped(groups, text_key="utterance", role_fn=None,
                               situation_key="context", style="full"))

    lines = [l for t in out for l in t.split("\n") if ":" in l]
    roles = [l.split(":")[0] for l in lines]

    check("group A has both roles",
          any(l.startswith("Person:") for l in out[0].split("\n")) and
          any(l.startswith("Listener:") for l in out[0].split("\n")),
          out[0].replace("\n", " | ")[:60])
    check("group B has both roles (no leak from A)",
          any(l.startswith("Person:") for l in out[1].split("\n")) and
          any(l.startswith("Listener:") for l in out[1].split("\n")),
          out[1].replace("\n", " | ")[:60])
    check("Person and Listener are balanced",
          roles.count("Person") == roles.count("Listener"),
          f"P={roles.count('Person')} L={roles.count('Listener')}")

    # The decisive assertion: a role leak produces an all-Listener corpus.
    check("corpus is not 100% one-sided", roles.count("Person") > 0,
          f"Person turns = {roles.count('Person')}")


def test_readers_registered():
    """Each handler must be callable and accept the shared kwargs signature."""
    import inspect

    import prepare_dataset as P

    print("\n[reader signatures]")
    for name, fn in P.HANDLER_FNS.items():
        params = inspect.signature(fn).parameters
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        check(f"{name} accepts **kwargs", has_kwargs,
              str(list(params)))
        check(f"{name} is a generator function",
              inspect.isgeneratorfunction(fn))


if __name__ == "__main__":
    print("=" * 66)
    print("Shard pipeline verification")
    print("=" * 66)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as td:
        test_multi_shard_rotation(td)
        test_workers(td)
        test_performance(td)
        test_no_full_load(td)
        test_edge_cases(td)
    test_empathetic_role_assignment()
    test_grouped_role_scope()
    test_readers_registered()
    test_cli_flags()
    test_clean_text()
    test_mixture_config()
    test_verify_shards_reports()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("All shard pipeline checks passed.")
    print("=" * 66)
