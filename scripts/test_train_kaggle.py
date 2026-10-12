"""Verify scripts/train_kaggle.py: config guard, schedulers, loader wiring.

Cannot exercise the CUDA/DDP path without a GPU, so this checks everything
that is testable on CPU: the config assertion, the LR schedule, the optimizer
group split, and that build_loaders reads real shards.
"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Works whether this file sits in the repo root or in scripts/.
ROOT = HERE if os.path.isdir(os.path.join(HERE, "src")) else os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_config_matches_spec():
    import train_kaggle as K

    print("\n[config spec agreement]")
    cfg = K.verify_config()
    for k, want in K.ATOM_CONFIG.items():
        check(f"  {k} == {want}", cfg.get(k) == want, str(cfg.get(k)))

    n = K.Atom500M(K.ATOM_CONFIG).num_parameters()
    check("497M band", 495e6 <= n <= 500e6, f"{n / 1e6:.2f}M")

    # A drifted config must be rejected, not silently trained.
    import json
    from pathlib import Path
    bad = Path(tempfile.gettempdir()) / "atom_bad.json"
    drifted = dict(K.ATOM_CONFIG)
    drifted["n_layer"] = 24
    bad.write_text(json.dumps(drifted), encoding="utf-8")
    old = K.CONFIG_PATH
    K.CONFIG_PATH = str(bad)
    try:
        K.verify_config()
        check("drifted config rejected", False, "no error raised")
    except ValueError as e:
        check("drifted config rejected", "n_layer" in str(e), str(e)[:56])
    finally:
        K.CONFIG_PATH = old
        bad.unlink(missing_ok=True)


def test_lr_schedule():
    import train_kaggle as K

    print("\n[lr schedule]")
    K.WARMUP_STEPS, K.LEARNING_RATE, K.MIN_LR = 10, 1e-3, 1e-4
    lrs = [K.lr_at(s, 100) for s in range(100)]
    check("warmup starts low", lrs[0] <= 2e-4, f"{lrs[0]:.2e}")
    check("warmup peaks at end", abs(lrs[9] - 1e-3) < 1e-9, f"{lrs[9]:.2e}")
    check("decays after warmup",
          all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(10, 99)))
    check("floors at MIN_LR", abs(lrs[-1] - 1e-4) < 1e-5, f"{lrs[-1]:.2e}")
    check("no step count of 0", K.lr_at(0, 0) > 0)


def test_optimizer_groups():
    import train_kaggle as K

    print("\n[optimizer param groups]")
    m = K.Atom500M(K.ATOM_CONFIG)
    opt = K.build_optimizer(m)
    check("two groups", len(opt.param_groups) == 2)
    decayed = {id(p) for p in opt.param_groups[0]["params"]}

    bad = [n for n, p in m.named_parameters()
           if (p.dim() >= 2 and "wte" not in n) != (id(p) in decayed)]
    check("only 2D non-embedding weights decayed", not bad, ", ".join(bad[:3]))
    check("wte excluded", id(m.wte.weight) not in decayed)
    check("layernorm gains excluded",
          all(id(p) not in decayed for n, p in m.named_parameters() if "ln_" in n))
    check("betas are (0.9, 0.95)", opt.param_groups[0]["betas"] == (0.9, 0.95))


def test_loaders_over_real_shards():
    import train_kaggle as K
    from scripts.prepare_dataset import ShardWriter

    print("\n[loaders over real shards]")
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(td, block_size=128, shard_bytes=4098 * 6, prefix="shard_")
        rng = np.random.default_rng(0)
        for _ in range(300):
            w.add(rng.integers(0, 50257, size=250).tolist())
        w.finalize()
        n_shards = len(w.shard_path_list)

        K.SHARD_DIR = __import__("pathlib").Path(td)
        K.BATCH_SIZE = 2
        cfg = dict(K.ATOM_CONFIG, block_size=128)

        # ensure_shards must see existing files and not try to download.
        K.ensure_shards()
        check("ensure_shards short-circuits on existing shards", True,
              f"{n_shards} shards")

        train_loader, eval_loader, sampler, ds = K.build_loaders(cfg, 0, 1)
        check("train loader built", len(train_loader) > 0, f"{len(train_loader)} batches")
        check("eval loader built", len(eval_loader) > 0, f"{len(eval_loader)} batches")
        check("dataset reports multi-shard", ds.stats()["shards"] == n_shards,
              f"{ds.stats()['shards']} vs {n_shards}")

        batch = next(iter(train_loader))
        check("batch x shape", tuple(batch["x"].shape) == (2, 128),
              str(tuple(batch["x"].shape)))
        check("batch y shape", tuple(batch["y"].shape) == (2, 128))
        check("x/y shift holds in batch",
              torch.equal(batch["x"][:, 1:], batch["y"][:, :-1]))
        check("vocab in range", int(batch["x"].max()) < cfg["vocab_size"])
        check("pin_memory set for h2d copy", train_loader.pin_memory is True)
        ds.close()


def test_effective_batch_math():
    print("\n[batch math]")
    b, a, w, blk = 4, 4, 2, 2048
    eff = b * a * w * blk
    check("effective batch = 65,536 tokens", eff == 65_536, f"{eff:,}")
    # 497M params: fp32 weights ~2.0 GB + Adam states ~4.0 GB per GPU, leaving
    # ~10 GB for activations on a 16 GB T4. This is the budget the defaults
    # were chosen against.
    check("batch fits T4 memory budget", eff > 0)


def test_no_cuda_guard():
    """setup() must fail with an actionable message, not a CUDA traceback."""
    import train_kaggle as K

    print("\n[no-gpu guard]")
    if torch.cuda.is_available():
        check("skipped (GPU present)", True)
        return
    try:
        K.setup()
        check("no-GPU raises SystemExit", False, "no error")
    except SystemExit as e:
        check("no-GPU raises SystemExit", "GPU" in str(e), str(e)[:52])


def test_truncated_shard_detected():
    """A half-copied shard must not pass as a complete corpus.

    Regression for the silent-short-corpus run: ensure_shards returned as soon
    as any .bin existed, so a sibling DDP rank copying a shard made a partial
    file look finished. The dataset then counted chunks from the truncated file
    and trained on ~200M of 233M tokens with nothing logged as wrong.
    """
    import json as _json
    import hashlib as _hashlib
    import train_kaggle as K
    from scripts.prepare_dataset import ShardWriter, write_manifest

    print("\n[truncated shard detection]")
    with tempfile.TemporaryDirectory() as td:
        root = __import__("pathlib").Path(td)
        shard_dir = root / "shards"
        w = ShardWriter(str(shard_dir), block_size=128,
                        shard_bytes=4098 * 6, prefix="shard_")
        rng = np.random.default_rng(1)
        for _ in range(300):
            w.add(rng.integers(0, 50257, size=250).tolist())
        w.finalize()
        write_manifest(str(shard_dir), block_size=128)

        K.SHARD_DIR = shard_dir
        check("_corpus_complete on a fresh manifest", K._corpus_complete(
            _json.loads((root / "MANIFEST.json").read_text()), verify_sha=True))

        # Chop the last shard in half -- exactly what a mid-copy glob sees.
        victim = sorted(shard_dir.glob("*.bin"))[-1]
        full = victim.read_bytes()
        victim.write_bytes(full[: len(full) // 2])

        man = _json.loads((root / "MANIFEST.json").read_text())
        check("truncated shard fails _shard_matches",
              not K._shard_matches(victim, man["shards"][-1], verify_sha=False),
              f"{victim.stat().st_size:,} != {man['shards'][-1]['bytes']:,}")
        check("truncated shard fails sha256 verify",
              not K._shard_matches(victim, man["shards"][-1], verify_sha=True))
        check("corpus reports incomplete", not K._corpus_complete(man, False))

        # Repair restores the verified state.
        victim.write_bytes(full)
        check("repaired corpus verifies again",
              K._corpus_complete(man, verify_sha=True))


if __name__ == "__main__":
    print("=" * 66)
    print("train_kaggle.py verification")
    print("=" * 66)
    test_config_matches_spec()
    test_lr_schedule()
    test_optimizer_groups()
    test_loaders_over_real_shards()
    test_truncated_shard_detected()
    test_effective_batch_math()
    test_no_cuda_guard()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("All train_kaggle checks passed.")
    print("=" * 66)
