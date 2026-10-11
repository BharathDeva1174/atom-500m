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
    # Successful uploads are NOT deleted by the uploader -- retention owns
    # deletion, so keep-2 actually retains two. A failed upload leaves its file
    # behind, which is the whole point of retaining a local fallback.
    check("no file deleted by the uploader itself",
          set(remaining) == {"step_1.pt", "bad.pt", "step_3.pt"},
          str(sorted(remaining)))


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

    print("\n[eval/checkpoint decoupling]")
    # Eval is 5x more frequent than checkpointing: an eval writes no files, so
    # a fine-grained loss curve is nearly free, while a checkpoint is ~5.6 GiB.
    check("EVAL_EVERY == 100", K.EVAL_EVERY == 100, str(K.EVAL_EVERY))
    check("SAVE_EVERY == 500", K.SAVE_EVERY == 500, str(K.SAVE_EVERY))
    check("cadences are decoupled", K.EVAL_EVERY != K.SAVE_EVERY,
          f"eval={K.EVAL_EVERY} save={K.SAVE_EVERY}")
    check("eval is the more frequent of the two",
          K.EVAL_EVERY < K.SAVE_EVERY, f"eval={K.EVAL_EVERY} save={K.SAVE_EVERY}")
    # 500 is a multiple of 100, so every checkpoint step is also an eval step:
    # decoupling costs no eval freshness at these values. This is a property of
    # the numbers, not a guarantee -- which is why eval_step is recorded.
    check("SAVE_EVERY is a multiple of EVAL_EVERY",
          K.SAVE_EVERY % K.EVAL_EVERY == 0,
          f"{K.SAVE_EVERY} % {K.EVAL_EVERY}")
    check("so checkpoint steps are also eval steps (gap 0)",
          K.SAVE_EVERY % K.EVAL_EVERY == 0,
          f"every {K.SAVE_EVERY}th step is a multiple of {K.EVAL_EVERY}")

    # Demonstrate the non-multiple case actually produces a gap, so the
    # eval_step field is not decorative. Asserted against computed values.
    def staleness(save_every: int, eval_every: int) -> int:
        """Largest gap between a checkpoint step and the nearest eval step."""
        s, worst = save_every, 0
        while s > 0:
            off = s % eval_every
            if off:
                worst = max(worst, eval_every - off)
            s -= eval_every
        return worst

    for save_every, want in ((500, 0), (250, 50), (300, 0), (700, 0),
                             (450, 50), (1000, 0)):
        got = staleness(save_every, K.EVAL_EVERY)
        check(f"  save={save_every}: staleness == {want}", got == want,
              f"got {got}")

    # The value actually recorded must make this recoverable: a checkpoint at
    # 250 with eval_every=100 carries eval_step=200, so eval_step - 0 is the
    # gap, not a silent 50.
    with tempfile.TemporaryDirectory() as td:
        from src.model import Atom500M
        from src.sync import save_local_checkpoint
        import torch as T
        m = Atom500M({"vocab_size": 512, "d_model": 64, "n_head": 4,
                      "n_layer": 2, "block_size": 32, "ff_expansion": 3.5})
        opt = T.optim.AdamW(m.parameters(), lr=1e-4)
        p = save_local_checkpoint(m, opt, step=250, config={}, loss=0.5,
                                 eval_loss=0.4, eval_step=200,
                                 checkpoint_dir=td)
        d = T.load(p, map_location="cpu", weights_only=False)
        check("stale checkpoint is self-describing",
              d["step"] - d["eval_step"] == 50,
              f"step={d['step']} eval_step={d['eval_step']} "
              f"gap={d['step'] - d['eval_step']}")

    import io as _io
    src = _io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "train_kaggle.py"), encoding="utf-8").read()
    i_eval = src.find("if step % EVAL_EVERY == 0:")
    i_save = src.find("if step % SAVE_EVERY == 0:")
    check("both cadence checks exist", i_eval > 0 and i_save > 0)
    # The cache must be assigned inside the eval branch and read inside the
    # checkpoint branch -- that is the whole mechanism.
    check("eval branch writes the cache",
          "latest_eval_loss = evaluate(" in src, "")
    check("cache is assigned with its source step",
          "latest_eval_step = step" in src, "")
    check("checkpoint branch reads the cache",
          "eval_loss=latest_eval_loss" in src, "")
    check("checkpoint records the eval's source step",
          "eval_step=latest_eval_step" in src, "")


def test_cached_eval_semantics():
    """A checkpoint must carry the cached eval, flagged with where it came from.

    With decoupled cadences the recorded eval is usually NOT from the
    checkpoint's own step. `eval_step` is what distinguishes a fresh
    measurement from a carried-over one, and omitting it would let a reader
    treat a 400-step-old number as current.
    """
    import torch as T

    from src.model import Atom500M
    from src.sync import save_local_checkpoint

    print("\n[cached eval semantics]")
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 32, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    opt = T.optim.AdamW(m.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as td:
        # Checkpoint at 500, most recent eval at 400.
        p = save_local_checkpoint(m, opt, step=500, config=cfg, loss=0.8,
                                 eval_loss=0.61, eval_step=400,
                                 checkpoint_dir=td)
        ck = T.load(p, map_location="cpu", weights_only=False)
        check("checkpoint step recorded", ck["step"] == 500, str(ck["step"]))
        check("cached eval_loss recorded", abs(ck["eval_loss"] - 0.61) < 1e-9,
              str(ck["eval_loss"]))
        check("eval_step records the eval's origin", ck["eval_step"] == 400,
              str(ck.get("eval_step")))
        check("eval_step is strictly before the checkpoint step",
              ck["eval_step"] < ck["step"],
              f"eval@{ck['eval_step']} ckpt@{ck['step']}")
        check("staleness is bounded by one interval",
              0 < ck["step"] - ck["eval_step"] <= 100,
              f"gap {ck['step'] - ck['eval_step']}")

        # No evaluation yet: the field must be present and null, not missing.
        p2 = save_local_checkpoint(m, opt, step=10, config=cfg, loss=0.9,
                                   checkpoint_dir=td)
        ck2 = T.load(p2, map_location="cpu", weights_only=False)
        check("eval_loss null before first eval", ck2["eval_loss"] is None,
              str(ck2["eval_loss"]))
        check("eval_step null before first eval", ck2["eval_step"] is None,
              str(ck2["eval_step"]))
        check("key present even when null", "eval_step" in ck2)


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


def test_retention_is_actually_effective():
    """Keep-2 must retain two files. Regression guard.

    Deleting on upload success and pruning by retention count were two
    independent policies over the same files. The delete won, because it ran
    first and left the pruner nothing to retain -- so KEEP_LOCAL_COUNT=2 was a
    silent no-op and a run kept no local fallback at all. These checks drive a
    real uploader through several checkpoints and assert files remain.
    """
    print("\n[retention is effective]")
    import train_kaggle as K

    class FakeApi:
        def __init__(self, token=None, **kw):
            self.calls = []

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id,
                        commit_message=None, **kw):
            self.calls.append(path_in_repo)

    import huggingface_hub
    real = huggingface_hub.HfApi
    # The worker does `from huggingface_hub import HfApi` *inside* the thread,
    # which re-resolves the attribute at call time -- so patching the module
    # attribute is both necessary and sufficient, as long as it is in place
    # before start().
    huggingface_hub.HfApi = FakeApi
    try:
        up = K.AsyncUploader("test/repo", token="x", max_pending=10)
        up.start()
        check("worker thread started", up._thread is not None
              and up._thread.is_alive(), "")
        check("uploader has no delete_after knob",
              not hasattr(up, "delete_after"),
              "delete-after-upload would override retention")

        with tempfile.TemporaryDirectory() as td:
            for step in (100, 200, 300, 400, 500, 600):
                p = os.path.join(td, f"step_{step}.pt")
                open(p, "wb").write(b"x")
                up.submit(p)
                # Wait for the upload to be confirmed before pruning. In a real
                # run the gap is ~500 optimizer steps, so uploads keep pace
                # trivially; submitting six instantly would legitimately leave
                # all six protected, which is correct behavior, not a bug.
                deadline = time.time() + 10
                while time.time() < deadline:
                    if os.path.abspath(p) in up.confirmed():
                        break
                    time.sleep(0.02)
                K.cleanup_stale_checkpoints(td, keep=K.KEEP_LOCAL_COUNT,
                                            protect=up.in_flight(td))
            up.close(timeout=20)
            left = sorted(os.listdir(td))

        check("all six were uploaded", up.uploaded == 6, str(up.uploaded))
        check("retention kept the expected number of files",
              len(left) == K.KEEP_LOCAL_COUNT,
              f"{len(left)} of 6 remain: {left}")
        check("kept files are the newest", left == ["step_500.pt", "step_600.pt"],
              str(left))
    finally:
        huggingface_hub.HfApi = real


def test_legacy_keep_local_alias():
    """ATOM_KEEP_LOCAL must still map onto the retention count."""
    print("\n[legacy ATOM_KEEP_LOCAL alias]")
    import importlib

    import train_kaggle as K

    def load(env):
        old = {k: os.environ.get(k) for k in env}
        for k in env:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            importlib.reload(K)
            return K.KEEP_LOCAL_COUNT
        finally:
            for k in env:
                os.environ.pop(k, None)
            if old:
                os.environ.update({k: v for k, v in old.items() if v is not None})
            importlib.reload(K)

    check("default is 2", load({}) == 2, str(load({})))
    check("ATOM_KEEP_LOCAL=1 -> keep all (0)", load({"ATOM_KEEP_LOCAL": "1"}) == 0,
          str(load({"ATOM_KEEP_LOCAL": "1"})))
    check("ATOM_KEEP_LOCAL=0 -> keep 1",
          load({"ATOM_KEEP_LOCAL": "0"}) == 1, str(load({"ATOM_KEEP_LOCAL": "0"})))
    check("explicit COUNT wins over legacy",
          load({"ATOM_KEEP_LOCAL": "1", "ATOM_KEEP_LOCAL_COUNT": "3"}) == 3,
          str(load({"ATOM_KEEP_LOCAL": "1", "ATOM_KEEP_LOCAL_COUNT": "3"})))
    check("COUNT=1 gives old delete-after-upload behavior",
          load({"ATOM_KEEP_LOCAL_COUNT": "1"}) == 1,
          str(load({"ATOM_KEEP_LOCAL_COUNT": "1"})))


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
    test_cached_eval_semantics()
    test_retention_is_actually_effective()
    test_legacy_keep_local_alias()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("All async-upload checks passed.")
    print("=" * 66)
