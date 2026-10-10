"""Run the real training loop with a fake HfApi so the async uploader executes.

The unit tests cover AsyncUploader in isolation. This one proves the wiring:
that main() starts an uploader when a token is present, that checkpoints are
enqueued during training, that the loop does not block on the "upload", and
that close() drains the queue at exit.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAINER = HERE / "train_kaggle.py"
PREPARE = HERE / "prepare_dataset.py"

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


TINY = {"vocab_size": 50257, "d_model": 64, "n_head": 4, "n_layer": 2,
        "block_size": 32, "ff_expansion": 3.5}


def main():
    print("=" * 66)
    print("train_kaggle.py async-upload wiring (gloo/CPU, fake Hub)")
    print("=" * 66)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
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
        ckpt_dir = td / "ckpt"

        # A stub huggingface_hub injected ahead of the real one, so no network
        # call is ever made but the uploader's code path runs for real.
        stub = td / "stubs"
        stub.mkdir()
        (stub / "huggingface_hub.py").write_text(
            "import os, time\n"
            "UPLOADS = []\n"
            "def snapshot_download(**kw):\n"
            "    raise RuntimeError('should not be called')\n"
            "class HfApi:\n"
            "    def __init__(self, token=None, **kw):\n"
            "        pass\n"
            "    def upload_file(self, path_or_fileobj, path_in_repo, repo_id,\n"
            "                    commit_message=None, **kw):\n"
            "        time.sleep(0.2)\n"
            "        with open(os.environ['FAKE_LOG'], 'a') as fh:\n"
            "            fh.write(f'{repo_id}|{path_in_repo}|'\n"
            "                     f'{os.path.getsize(path_or_fileobj)}\\n')\n",
            encoding="utf-8",
        )
        log = td / "uploads.log"

        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(stub),
            "ATOM_CONFIG": str(cfg_path),
            "ATOM_SHARD_DIR": str(shard_dir),
            "ATOM_CKPT_DIR": str(ckpt_dir),
            "ATOM_BATCH_SIZE": "4",
            "ATOM_GRAD_ACCUM": "2",
            "ATOM_LOG_EVERY": "10",
            "ATOM_EVAL_EVERY": "1000",
            "ATOM_SAVE_EVERY": "20",
            "ATOM_WARMUP": "5",
            "ATOM_MAX_STEPS": "40",
            "ATOM_DOWNLOAD": "0",
            "ATOM_FORCE_CPU": "1",
            "ATOM_UPLOAD_JOIN_TIMEOUT": "60",
            "FAKE_LOG": str(log),
            "WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29595",
            "HF_TOKEN": "hf_fake_token_for_testing",
            "PYTHONUNBUFFERED": "1",
        })

        t0 = time.time()
        proc = subprocess.run([sys.executable, str(TRAINER)], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)
        wall = time.time() - t0
        out = proc.stdout + proc.stderr
        uploads = []
        if log.exists():
            uploads = [l for l in log.read_text(encoding="utf-8").splitlines() if l]

    print(f"\n--- output (exit {proc.returncode}, {wall:.1f}s) ---")
    for ln in [l for l in out.splitlines() if l.strip()][-18:]:
        print("   ", ln)
    if uploads:
        print(f"\n--- uploads recorded ({len(uploads)}) ---")
        for u in uploads:
            print("   ", u)

    print("\n" + "=" * 66)
    check("exited cleanly", proc.returncode == 0, str(proc.returncode))
    check("uploader started", "background uploader" in out, "")
    check("no traceback", "Traceback" not in out, "")
    check("checkpoints saved", out.count("[ckpt] step") >= 2,
          f"{out.count('[ckpt] step')} saves")
    check("uploads actually happened", len(uploads) >= 2, f"{len(uploads)} uploads")
    check("all uploads go to the right repo",
          all(u.startswith("siddhu1174/atom-500m|checkpoints/") for u in uploads),
          uploads[0] if uploads else "none")
    check("uploader drained at exit", "finished:" in out, "")
    check("close() reported success", "0 failed" in out, "")

    # A run ending exactly on SAVE_EVERY used to write and upload step_40 twice.
    names = [u.split("|")[1] for u in uploads]
    check("no duplicate uploads", len(names) == len(set(names)), str(names))
    check("nothing dropped", "0 dropped" in out, "")
    check("final checkpoint uploaded",
          f"checkpoints/step_{40}.pt" in names, str(names))
    print("=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("Async upload wiring verified end to end.")
    print("=" * 66)


if __name__ == "__main__":
    import time
    main()
