"""Verify AsyncUploader: non-blocking, bounded, failure-tolerant, drains on close.

The point of the background uploader is that the training loop never waits on
the network. These tests measure that directly rather than asserting it.
"""
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_non_blocking():
    """submit() must return fast even when the upload is slow."""
    import train_kaggle as K

    print("\n[non-blocking submit]")
    up = K.AsyncUploader("test/repo", token="x", max_pending=5)
    # Replace the worker with one that blocks for a long time, simulating a
    # slow connection. If submit() stalls, training would stall too.
    up._thread = threading.Thread(target=lambda: time.sleep(30), daemon=True)
    up._thread.start()

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            p = os.path.join(td, f"step_{i}.pt")
            open(p, "wb").write(b"x")
            paths.append(p)

        t0 = time.time()
        for p in paths:
            up.submit(p)
        elapsed = time.time() - t0
    check("3 submits took < 50ms", elapsed < 0.05, f"{elapsed * 1000:.1f}ms")
    check("all 3 queued while worker is blocked", up.pending() == 3,
          f"pending={up.pending()}")


def test_queue_bound():
    """A full queue must drop old files, not grow without bound."""
    import train_kaggle as K

    print("\n[bounded queue]")
    up = K.AsyncUploader("test/repo", token="x", max_pending=2)
    up._thread = threading.Thread(target=lambda: time.sleep(20), daemon=True)
    up._thread.start()

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(6):
            p = os.path.join(td, f"step_{i}.pt")
            open(p, "wb").write(b"x")
            paths.append(p)
        for p in paths:
            up.submit(p)
        check("queue never exceeds max_pending", up.pending() <= 2,
              f"pending={up.pending()} max=2")
        check("dropped counted", up.dropped >= 3, f"dropped={up.dropped}")
        alive = [p for p in paths if os.path.exists(p)]
        check("dropped files deleted from disk", len(alive) <= 3,
              f"{len(alive)} of 6 remain")


def test_worker_resilience():
    """An upload error must not kill the thread or stop later uploads."""
    import train_kaggle as K

    print("\n[worker survives failures]")

    class FakeApi:
        def __init__(self):
            self.calls = []

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id,
                        commit_message=None):
            self.calls.append(path_in_repo)
            if "bad" in path_in_repo:
                raise RuntimeError("simulated network failure")
            return None

    import huggingface_hub
    real = huggingface_hub.HfApi
    huggingface_hub.HfApi = lambda token=None: FakeApi()
    try:
        up = K.AsyncUploader("test/repo", token="x", max_pending=10)
        up.start()
        with tempfile.TemporaryDirectory() as td:
            for name in ("step_1.pt", "bad.pt", "step_3.pt"):
                p = os.path.join(td, name)
                open(p, "wb").write(b"x")
                up.submit(p)
            up.close(timeout=10)
            remaining = os.listdir(td)
    finally:
        huggingface_hub.HfApi = real

    check("failure counted", up.failed == 1, f"failed={up.failed}")
    check("good uploads succeeded", up.uploaded == 2, f"uploaded={up.uploaded}")
    check("uploaded files deleted after success",
          set(remaining) == {"bad.pt"}, str(remaining))


def test_close_drains():
    """close() must finish queued uploads, not abandon them."""
    import train_kaggle as K

    print("\n[close drains queue]")

    class FakeApi:
        def __init__(self):
            self.calls = []

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id,
                        commit_message=None):
            time.sleep(0.05)
            self.calls.append(path_in_repo)

    import huggingface_hub
    api = FakeApi()
    real = huggingface_hub.HfApi
    huggingface_hub.HfApi = lambda token=None: api
    try:
        up = K.AsyncUploader("test/repo", token="x", max_pending=10)
        up.start()
        with tempfile.TemporaryDirectory() as td:
            for i in range(4):
                p = os.path.join(td, f"step_{i}.pt")
                open(p, "wb").write(b"x")
                up.submit(p)
            up.close(timeout=30)
    finally:
        huggingface_hub.HfApi = real

    check("all 4 uploaded before close returned", up.uploaded == 4,
          f"uploaded={up.uploaded}")
    check("queue empty after close", up.pending() == 0, f"pending={up.pending()}")
    check("no duplicates", len(api.calls) == len(set(api.calls)), str(api.calls))


def test_close_timeout():
    """A hung upload must not hang close() forever."""
    import train_kaggle as K

    print("\n[close timeout]")

    class HangApi:
        def upload_file(self, **kw):
            time.sleep(60)

    import huggingface_hub
    real = huggingface_hub.HfApi
    huggingface_hub.HfApi = lambda token=None: HangApi()
    try:
        up = K.AsyncUploader("test/repo", token="x", max_pending=10)
        up.start()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "step_1.pt")
            open(p, "wb").write(b"x")
            up.submit(p)
            t0 = time.time()
            up.close(timeout=1.0)
            elapsed = time.time() - t0
    finally:
        huggingface_hub.HfApi = real
    check("close returns despite hung upload", elapsed < 10,
          f"{elapsed:.1f}s with a 1s timeout")


def test_checkpoint_contents():
    """A saved checkpoint must carry step, model, optimizer, and loss."""
    import torch

    import train_kaggle as K
    from src.model import Atom500M
    from src.sync import load_checkpoint, save_local_checkpoint

    print("\n[checkpoint contents]")
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 32, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def T_load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    with tempfile.TemporaryDirectory() as td:
        path = save_local_checkpoint(m, opt, step=7, config=cfg, loss=1.2345,
                                     checkpoint_dir=td)
        check("file written", path is not None and os.path.exists(path))

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("step", "model", "optimizer", "config", "loss"):
            check(f"  has '{key}'", key in ckpt)
        check("step recorded", ckpt["step"] == 7, str(ckpt["step"]))
        check("loss recorded", abs(ckpt["loss"] - 1.2345) < 1e-9,
              str(ckpt["loss"]))
        check("model state is a state dict", "wte.weight" in ckpt["model"])
        check("optimizer state present", ckpt["optimizer"] is not None)
        # The tie must survive serialization. A naive {k: v.cpu()} would split
        # it into two copies -- +206 MB per checkpoint, and a model that loses
        # tying on resume. Storage identity is the real test.
        check("checkpoint preserves the weight tie",
              ckpt["model"]["lm_head.weight"].untyped_storage().data_ptr()
              == ckpt["model"]["wte.weight"].untyped_storage().data_ptr())

        # eval_loss rides along so a resumed run can compare against the same
        # held-out measurement the producing run recorded.
        p2 = save_local_checkpoint(m, opt, step=8, config=cfg, loss=0.9,
                                   eval_loss=0.75, checkpoint_dir=td)
        ck2 = T_load(p2)
        check("eval_loss recorded", abs(ck2["eval_loss"] - 0.75) < 1e-9,
              str(ck2.get("eval_loss")))
        check("loss and eval_loss are separate fields",
              abs(ck2["loss"] - 0.9) < 1e-9, str(ck2.get("loss")))


def test_eval_save_alignment():
    """EVAL_EVERY and SAVE_EVERY should coincide, by design.

    Aligned, each checkpoint interval carries exactly one eval, and the eval
    loss measured at that step is available to the checkpoint written right
    after it. Misaligned, the recorded eval is stale by up to a full interval.
    """
    import train_kaggle as K

    print("\n[eval/checkpoint alignment]")
    check("EVAL_EVERY == SAVE_EVERY", K.EVAL_EVERY == K.SAVE_EVERY,
          f"eval={K.EVAL_EVERY} save={K.SAVE_EVERY}")
    check("both are 500 by default", K.EVAL_EVERY == 500 and K.SAVE_EVERY == 500,
          f"eval={K.EVAL_EVERY} save={K.SAVE_EVERY}")

    # The loop must run eval before checkpoint on a shared step, otherwise the
    # checkpoint records the previous interval's eval.
    import io as _io
    src = _io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "train_kaggle.py"), encoding="utf-8").read()
    i_eval = src.find("if step % EVAL_EVERY == 0:")
    i_save = src.find("if step % SAVE_EVERY == 0:")
    check("both cadence checks exist", i_eval > 0 and i_save > 0)
    check("eval runs before checkpoint in the loop", 0 < i_eval < i_save,
          f"eval@{i_eval} save@{i_save}")
    check("checkpoint receives the eval loss",
          "eval_loss=last_eval" in src)


def test_round_trip_into_model():
    """The saved checkpoint must load back into a live model unchanged."""
    import torch as T

    import train_kaggle as K
    from src.model import Atom500M
    from src.sync import load_checkpoint, save_local_checkpoint

    print("\n[checkpoint round trip]")
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 32, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    opt = T.optim.AdamW(m.parameters(), lr=1e-4)
    # Train briefly so the saved weights are not the initial ones.
    for _ in range(3):
        idx = T.randint(0, 512, (2, 16))
        m(idx, T.randint(0, 512, (2, 16))).backward()
        opt.step()
        opt.zero_grad()

    with tempfile.TemporaryDirectory() as td:
        p = save_local_checkpoint(m, opt, step=7, config=cfg, loss=1.2345,
                                 eval_loss=0.99, checkpoint_dir=td)
        check("file written", p is not None and os.path.exists(p))

        # And it round-trips back into a live model.
        m2 = Atom500M(cfg)
        step = load_checkpoint(p, m2, opt)
        check("reload restores step", step == 7, str(step))
        # Assert the tie by shared storage, not object identity:
        # `state_dict()` returns fresh `detach()` views, so `is` is False even
        # for genuinely tied weights. Storage identity is what actually matters
        # -- a split tie would mean a duplicated 51M-param tensor on resume.
        check("module weights tied after reload", m2.lm_head.weight is m2.wte.weight)
        sd = m2.state_dict()
        check("state_dict shares storage after reload",
              sd["lm_head.weight"].untyped_storage().data_ptr()
              == sd["wte.weight"].untyped_storage().data_ptr())


if __name__ == "__main__":
    print("=" * 66)
    print("AsyncUploader verification")
    print("=" * 66)
    test_non_blocking()
    test_queue_bound()
    test_worker_resilience()
    test_close_drains()
    test_close_timeout()
    test_checkpoint_contents()
    test_round_trip_into_model()
    test_eval_save_alignment()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("All async-upload checks passed.")
    print("=" * 66)
