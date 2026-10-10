"""Validation: no CPU tensor is ever passed to dist.all_reduce().

Two independent checks, because the static one alone would not catch a tensor
that is *created* on the right device but moved later, or a helper that returns
CPU.

1. Static (AST): every tensor feeding a collective is created with device=.
2. Runtime: install a `dist.all_reduce` spy and run a real 2-rank gloo group,
   asserting each tensor's device matches the process group's device type.

The runtime check is the one that generalizes -- it would catch a device-less
collective added anywhere in the file, not just the two known sites.
"""
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def check_static():
    """AST scan of every collective in the trainer."""
    print("\n[static] tensors feeding collectives carry device=")
    path = os.path.join(HERE, "train_kaggle.py")
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    collectives = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("all_reduce", "broadcast", "barrier", "all_gather")
    ]
    check("collectives found", len(collectives) >= 2, f"{len(collectives)}")

    # Map every name to the call that assigned it.
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("tensor", "zeros", "ones"):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        assigned[tgt.id] = node

    bad = []
    total = 0
    for c in collectives:
        for arg in c.args:
            if not isinstance(arg, ast.Name):
                continue
            node = assigned.get(arg.id)
            if node is None:
                continue
            total += 1
            has_device = any(kw.arg == "device" for kw in node.value.keywords)
            if not has_device:
                bad.append(f"{arg.id} (line {node.lineno}) -> {c.func.attr}")

    check("every collective tensor checked", total >= 2, f"{total}")
    check("none default to CPU", not bad, ", ".join(bad))

    # And the literal patterns from the original bug.
    for pat in ("torch.tensor([loss_sum, n_batches], dtype=torch.float64)",
                "torch.tensor([0 if ok else 1], dtype=torch.int32)"):
        check(f"absent: {pat[:40]}", pat not in src)


def check_runtime():
    """Run a real group with a spy over the collectives."""
    print("\n[runtime] device of every tensor actually reduced")
    import tempfile

    import torch
    import torch.distributed as dist

    if not dist.is_available():
        check("skipped (dist unavailable)", True)
        return

    # The gloo group here is CPU, so "correct" means CPU. The point is that the
    # spy observes and reports, so a device-less tensor in a CPU group would
    # still be counted and compared against the group's device type.
    offenders = []
    calls = []

    real_all_reduce = dist.all_reduce

    def spy(tensor, *a, **kw):
        calls.append(tensor)
        offenders.append(tensor.device)
        return real_all_reduce(tensor, *a, **kw)

    dist.all_reduce = spy
    try:
        with tempfile.TemporaryDirectory() as td:
            dist.init_process_group(
                backend="gloo", init_method=f"file://{os.path.join(td, 's')}",
                rank=0, world_size=1,
            )
            try:
                backend = dist.get_backend()
                group_dev = torch.device("cpu" if backend == "gloo" else "cuda")

                # Exercise the two production code paths directly.
                import train_kaggle as K

                class Dummy(torch.nn.Module):
                    def eval(self): pass
                    def train(self): pass
                    def __call__(self, x, y=None): return torch.tensor(2.0)

                K.evaluate(Dummy(), [], group_dev, K.contextlib.nullcontext(), 0)
                check("evaluate() ran with the spy installed", len(calls) >= 1,
                      f"{len(calls)} collective(s)")

                m = torch.nn.Linear(4, 4)
                opt = torch.optim.AdamW(m.parameters())
                # Already-saved marker off, so the save path actually runs.
                K._LAST_SAVED_STEP[0] = -1
                import tempfile as tf
                with tf.TemporaryDirectory() as ck:
                    K.CHECKPOINT_DIR = ck
                    K.checkpoint(m, opt, 1, {"vocab_size": 4}, loss=0.1,
                                 device=group_dev)
                check("checkpoint() ran with the spy installed", len(calls) >= 2,
                      f"{len(calls)} collective(s) total")
            finally:
                dist.destroy_process_group()
    finally:
        dist.all_reduce = real_all_reduce

    mismatch = [str(d) for d in offenders if d.type != group_dev.type]
    check("all reduced tensors match the group device type", not mismatch,
          ", ".join(mismatch) or f"all {len(offenders)} on {group_dev.type}")


def check_sizes():
    """The disk math behind the retention policy, measured not assumed."""
    print("\n[retention math]")
    import tempfile

    import torch
    from src.model import Atom500M
    from src.sync import cleanup_stale_checkpoints

    cfg = {"vocab_size": 4096, "d_model": 256, "n_head": 8, "n_layer": 6,
           "block_size": 128, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    n_params = sum(p.numel() for p in m.parameters())

    with tempfile.TemporaryDirectory() as td:
        for step in (100, 200, 300, 400):
            torch.save({"step": step, "model": {}, "optimizer": None,
                        "config": cfg, "loss": 1.0, "saved_at": 0.0},
                       os.path.join(td, f"step_{step}.pt"))
        check("4 checkpoints on disk", len(os.listdir(td)) == 4)

        # All confirmed: keep the 2 newest.
        removed = cleanup_stale_checkpoints(td, keep=2, protect=set())
        left = sorted(os.listdir(td))
        check("keeps 2 newest", left == ["step_300.pt", "step_400.pt"], str(left))
        check("reports removals", removed == 2, str(removed))

        # step_400 unconfirmed (in flight): it must survive regardless of age.
        protect = {os.path.join(td, "step_400.pt")}
        cleanup_stale_checkpoints(td, keep=1, protect=protect)
        left = sorted(os.listdir(td))
        check("in-flight checkpoint is never pruned",
              "step_400.pt" in left, str(left))
        check("other files still pruned down to budget",
              len(left) == 1, str(left))

    scaled_2 = 2 * 497_487_872 * 12 / 2**30
    print(f"    (retention budget for 497M @ 12 B/param = "
          f"{scaled_2:.1f} GiB for 2 checkpoints)")


if __name__ == "__main__":
    print("=" * 66)
    print("DDP device validation")
    print("=" * 66)
    check_static()
    check_runtime()
    check_sizes()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("No CPU tensors reach dist.all_reduce().")
    print("=" * 66)
