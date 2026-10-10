"""Exercise train_kaggle.py's actual training loop on CPU with a gloo group.

The DDP/NCCL path needs real NVIDIA hardware, but everything else -- the
accumulation boundary, the unscale->clip->step ordering, the checkpoint
barrier, the LR schedule, eval all-reduce -- is device-independent. Running the
real `main()` under a 2-rank gloo group with a tiny config exercises those.

The CUDA-specific pieces (autocast fp16, GradScaler) are substituted, and the
substitution is logged rather than hidden.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Works whether this file sits in the repo root or in scripts/.
ROOT = HERE if os.path.isdir(os.path.join(HERE, "src")) else os.path.dirname(HERE)
TRAINER = os.path.join(ROOT, "scripts", "train_kaggle.py")
PREPARE = os.path.join(ROOT, "scripts", "prepare_dataset.py")

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


TINY = {
    "vocab_size": 50257,   # real gpt2 vocab: the corpus is really tokenized
    "d_model": 64, "n_head": 4, "n_layer": 2,
    "block_size": 32, "ff_expansion": 3.5,
}


def provision(td: Path):
    """Tiny config + real shards, packed by the real prepare_dataset."""
    cfg_path = td / "config.json"
    cfg_path.write_text(json.dumps(TINY), encoding="utf-8")

    raw = td / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as fh:
        for i in range(3000):
            fh.write(json.dumps({"text": f"doc {i} " + "hello " * 30}) + "\n")

    shard_dir = td / "shards"
    subprocess.run(
        [sys.executable, str(PREPARE), "--input", str(raw),
         "--out_dir", str(shard_dir), "--block_size", "32",
         "--shard_bytes", str(33 * 1024), "--progress_every", "0"],
        check=True, capture_output=True,
    )
    n = len(list(shard_dir.glob("*.bin")))
    return cfg_path, shard_dir, n


def main():
    print("=" * 66)
    print("train_kaggle.py loop verification (gloo/CPU)")
    print("=" * 66)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg_path, shard_dir, n_shards = provision(td)
        print(f"[fixtures] {n_shards} shards in {shard_dir.name}")
        store = td / "rdzv"
        env = dict(os.environ)
        env.update({
            "ATOM_CONFIG": str(cfg_path),
            "ATOM_SHARD_DIR": str(shard_dir),
            "ATOM_CKPT_DIR": str(td / "ckpt"),
            "ATOM_BATCH_SIZE": "4",
            "ATOM_GRAD_ACCUM": "2",
            "ATOM_LOG_EVERY": "10",
            "ATOM_EVAL_EVERY": "20",
            "ATOM_SAVE_EVERY": "25",
            "ATOM_WARMUP": "5",
            "ATOM_MAX_STEPS": "60",
            "ATOM_DOWNLOAD": "0",
            "ATOM_FORCE_CPU": "1",
            "WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29591",
            "HF_TOKEN": "",  # never touch the network from a test
            "ATOM_INIT_METHOD": f"file://{store}",
            "PYTHONUNBUFFERED": "1",
        })

        procs = []
        for rank in range(2):
            e = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
            procs.append(subprocess.Popen(
                [sys.executable, TRAINER], env=e, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))

        outs = []
        for p in procs:
            try:
                o, _ = p.communicate(timeout=420)
            except subprocess.TimeoutExpired:
                p.kill()
                o, _ = p.communicate()
                o = (o or "") + "\n*** TIMED OUT ***"
            outs.append(o)
        codes = [p.returncode for p in procs]

    print(f"\n--- rank 0 output (exit {codes[0]}) " + "-" * 30)
    for ln in [x for x in outs[0].splitlines() if x.strip()][-24:]:
        print("   ", ln)

    combined = "\n".join(outs)
    print("\n" + "=" * 66)
    checks = [
        ("both ranks exited cleanly", all(c == 0 for c in codes), str(codes)),
        ("tiny config accepted on CPU harness", "parameters" in combined, ""),
        ("shards loaded", '"shards":' in combined, ""),
        ("all shards used", f'"shards": {n_shards}' in combined, str(n_shards)),
        ("optimizer groups built", "decayed" in combined, ""),
        ("effective batch logged", "effective batch" in combined, ""),
        ("training stepped", "step " in combined and "loss" in combined, ""),
        ("eval ran", "eval " in combined, ""),
        ("checkpoint saved", "[sync] saved" in combined or "[ckpt] step" in combined, ""),
        ("no token -> no uploader", "no background upload" in combined, ""),
        ("process group destroyed", "process group destroyed" in combined, ""),
        ("no traceback", "Traceback" not in combined, ""),
        ("completed without timeout", "TIMED OUT" not in combined, ""),
    ]
    for name, ok, detail in checks:
        check(name, ok, detail)

    print("=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("train_kaggle training loop verified end to end.")
    print("=" * 66)


if __name__ == "__main__":
    main()
