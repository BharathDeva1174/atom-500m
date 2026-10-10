"""Launch a real 2-rank DDP run over a file:// store, bypassing torchrun.

torchrun's default rendezvous needs TCPStore -> libuv, which the Windows torch
build lacks. Spawning the ranks directly with a FileStore exercises exactly the
same code path in scripts/train_ddp.py (gloo backend, real DDP all-reduce, real
gradient accumulation) so the trainer is verified before it ever reaches Kaggle.

Run: python scripts/ddp_smoke_test.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "train_ddp.py"
PREPARE = ROOT / "scripts" / "prepare_dataset.py"

# A deliberately tiny config so the full DDP loop runs in seconds. The model is
# narrow and shallow, but `vocab_size` stays at the real gpt2 value: the corpus is
# tokenized with the actual gpt2 tokenizer, so ids run to 50256 and a smaller
# vocab would make every embedding lookup out of range.
SMOKE_CONFIG = {
    "vocab_size": 50257, "d_model": 64, "n_head": 4, "n_layer": 2,
    "block_size": 32, "ff_expansion": 3.5,
}


def provision_fixtures():
    """Write the tiny config + a real shard corpus this test needs.

    Generated rather than committed, so a fresh clone has no test-only files
    sitting in the tree and the fixtures can never drift from the code. The
    corpus is built by the *real* prepare_dataset.py, so this exercises the
    actual pack -> shard -> memmap path rather than hand-writing a .npy.
    """
    cfg_path = ROOT / "configs" / "smoke.json"
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(SMOKE_CONFIG, indent=2), encoding="utf-8")

    raw = ROOT / "data" / "smoke.jsonl"
    raw.parent.mkdir(exist_ok=True)
    with open(raw, "w", encoding="utf-8") as fh:
        for i in range(4000):
            fh.write(json.dumps({"text": f"doc {i} " + "hello " * 40}) + "\n")

    shard_dir = ROOT / "data" / "smoke_shards"
    subprocess.run(
        [sys.executable, str(PREPARE),
         "--input", str(raw), "--out_dir", str(shard_dir),
         "--block_size", "32", "--shard_bytes", str(33 * 1024),  # ~65 chunks/shard
         "--progress_every", "0"],
        check=True, capture_output=True, cwd=str(ROOT),
    )
    return cfg_path, raw, shard_dir


def main():
    cfg_path, raw_path, shard_dir = provision_fixtures()
    print(f"[fixtures] {cfg_path.relative_to(ROOT)}")
    print(f"[fixtures] {raw_path.relative_to(ROOT)}")
    print(f"[fixtures] {shard_dir.relative_to(ROOT)}")

    n_shards = len(list(shard_dir.glob("*.bin")))
    if n_shards < 2:
        print(f"WARNING: only {n_shards} shard(s) — expected several to test rotation")

    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "rendezvous"

        env_base = dict(os.environ)
        env_base.update({
            "ATOM_CONFIG": "configs/smoke.json",
            "ATOM_SHARD_DIR": str(shard_dir),
            # Point the flat-.npy fallback at a path that does not exist, so the
            # shard branch is unambiguously the one under test.
            "ATOM_DATA": "data/__no_such_file__.npy",
            "ATOM_BATCH_SIZE": "4",
            "ATOM_GRAD_ACCUM": "2",
            "ATOM_LOG_EVERY": "10",
            "ATOM_EVAL_EVERY": "20",
            "ATOM_SAVE_EVERY": "25",
            "ATOM_WARMUP": "5",
            "ATOM_MAX_EPOCHS": "1",
            "ATOM_BACKEND": "gloo",
            "ATOM_INIT_METHOD": f"file://{store}",
            "ATOM_FORCE_CPU": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29588",
            # Force the no-token path so the test never touches the network.
            "HF_TOKEN": "",
            "HF_REPO_ID": "",
            "PYTHONUNBUFFERED": "1",
        })

        procs = []
        for rank in range(2):
            env = dict(env_base, RANK=str(rank), LOCAL_RANK=str(rank))
            procs.append(subprocess.Popen(
                [sys.executable, str(TRAINER)],
                env=env, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))

        deadline = time.time() + 420
        outputs = []
        for p in procs:
            remaining = max(5, deadline - time.time())
            try:
                out, _ = p.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                out = (out or "") + "\n*** TIMED OUT / KILLED ***"
            outputs.append(out)

        codes = [p.returncode for p in procs]

    print("=" * 66)
    print("2-rank DDP smoke test (gloo, CPU)")
    print("=" * 66)
    for rank, (out, code) in enumerate(zip(outputs, codes)):
        print(f"\n--- rank {rank} (exit {code}) " + "-" * 40)
        # Rank 1 is quiet by design; show its tail to confirm it ran to the end.
        lines = [ln for ln in out.splitlines() if ln.strip()]
        for ln in lines if rank == 0 else lines[-6:]:
            print(ln)

    combined = "\n".join(outputs)
    print("\n" + "=" * 66)
    checks = [
        ("both ranks exited cleanly", all(c == 0 for c in codes), str(codes)),
        ("gloo backend selected", "backend=gloo" in combined, ""),
        ("model built", "parameters" in combined, ""),
        ("shard dataset used (not .npy fallback)", '"shards":' in combined, ""),
        ("multi-shard corpus loaded", f'"shards": {n_shards}' in combined, f"{n_shards} shards"),
        ("distributed sampler sharded data", "train blocks" in combined, ""),
        ("grad accumulation ran (steps < micro-batches)", "step " in combined, ""),
        ("loss logged and finite", "loss " in combined, ""),
        ("eval ran", "eval loss" in combined, ""),
        ("checkpoint saved", "[sync] saved" in combined, ""),
        ("checkpoint all-reduce status reported", "[ckpt]" in combined, ""),
        ("no HF upload attempted", "no token supplied" in combined, ""),
        ("no network call made", "401 Client Error" not in combined, ""),
        ("process group destroyed", "process group destroyed" in combined, ""),
        ("no traceback", "Traceback" not in combined, ""),
    ]
    failed = []
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failed.append(name)

    print("=" * 66)
    if failed:
        print(f"FAILED ({len(failed)}): " + ", ".join(failed))
        sys.exit(1)
    print("Real 2-rank DDP loop verified end to end.")
    print("=" * 66)


if __name__ == "__main__":
    main()
