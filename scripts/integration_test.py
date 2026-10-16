"""End-to-end integration test on CPU. Not part of the Kaggle flow — this is
the local harness that exercises the real training-step code path (autocast
substituted for fp16, DDP stubbed) so a broken trainer is caught here instead
of after a 40-minute Kaggle clone.

Run: python scripts/integration_test.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import PretokenizedDataset, collate  # noqa: E402
from src.model import Atom500M  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_tiny_training_loop():
    """Mimic train_ddp.py's inner loop: accum -> unscale -> clip -> step -> update."""
    torch.manual_seed(0)
    cfg = {
        "vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
        "block_size": 32, "ff_expansion": 3.5,
    }
    model = Atom500M(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # no fp16 on CPU

    # A learnable pattern the model should be able to fit: ids cycle mod vocab.
    data = (np.arange(4000) % 512).astype(np.uint16)
    ds = PretokenizedDataset(data, block_size=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate, drop_last=True)

    losses = []
    model.train()
    for epoch in range(30):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for it, b in enumerate(loader):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=False):
                loss = model(b["x"], b["y"])
            scaler.scale(loss / 2).backward()
            running += loss.item()
            if (it + 1) % 2 != 0:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        losses.append(running / max(1, len(loader)))

    check("loss decreases over 30 epochs", losses[-1] < losses[0],
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    check("final loss below ln(V)=6.24", losses[-1] < np.log(512),
          f"{losses[-1]:.3f}")
    check("loss is finite throughout", all(np.isfinite(l) for l in losses))
    check("gradients actually flowed", model.wte.weight.grad is not None)
    return model


def test_tied_weight_gradients(model):
    """Tied weights must receive the summed gradient, not one copy's."""
    g = model.wte.weight.grad
    check("wte.weight has gradient", g is not None)
    check("wte.weight gradient is nonzero", g is not None and g.abs().sum() > 0,
          f"|g| = {g.abs().sum():.4f}" if g is not None else "")


def test_generate():
    """generate() must respect block_size and produce the requested length."""
    cfg = {
        "vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
        "block_size": 32, "ff_expansion": 3.5,
    }
    m = Atom500M(cfg).eval()
    idx = torch.randint(0, 512, (2, 8))
    out = m.generate(idx, max_new_tokens=12, temperature=0.0)
    check("generate returns right shape", tuple(out.shape) == (2, 20), str(tuple(out.shape)))
    check("generate preserves the prompt", torch.equal(out[:, :8], idx))

    # A prompt that leaves exactly enough room must still generate.
    fits = torch.randint(0, 512, (1, 29))  # 29 + 3 == block_size 32
    try:
        check("generate fills context exactly", tuple(m.generate(
            fits, max_new_tokens=3, temperature=0.0).shape) == (1, 32))
    except Exception as e:
        check("generate fills context exactly", False, str(e))

    # One token too many: absolute positions cannot wrap, so this must fail
    # fast with an actionable message rather than mid-generation.
    full = torch.randint(0, 512, (1, 32))
    try:
        m.generate(full, max_new_tokens=3, temperature=0.0)
        check("generate rejects over-long request", False, "no error raised")
    except ValueError as e:
        check("generate rejects over-long request", "block_size" in str(e),
              str(e)[:60])


def test_positional_embeddings():
    """The model must be position-sensitive, not permutation-equivariant.

    This is the regression test for the missing-wpe bug: without positional
    embeddings, causal self-attention computes the *same* representation for
    every reordering of a token sequence, so "dog bites man" and "man bites dog"
    are indistinguishable. For an empathy model that reads conversational turn
    order, that is a fatal defect rather than a stylistic choice.
    """
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 32, "ff_expansion": 3.5}
    m = Atom500M(cfg).eval()

    check("wpe exists", hasattr(m, "wpe") and m.wpe.num_embeddings == 32,
          f"shape {tuple(m.wpe.weight.shape)}")
    check("wpe is a trainable parameter",
          any(p is m.wpe.weight for p in m.parameters()))

    with torch.no_grad():
        a = torch.tensor([[10, 20, 30, 40]])
        b = torch.tensor([[40, 30, 20, 10]])
        d_perm = (m(a) - m(b)).abs().max().item()
    check("permuting tokens changes the output", d_perm > 1e-6, f"drift {d_perm:.4f}")

    # start_pos must actually move the encoding — otherwise generate() slides
    # its window while silently re-reading every token as position 0.
    x = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        d_pos = (m(x, start_pos=0) - m(x, start_pos=7)).abs().max().item()
    check("start_pos shifts the encoding", d_pos > 1e-6, f"drift {d_pos:.4f}")

    # A batch must not share state across the start_pos default.
    with torch.no_grad():
        eq = torch.allclose(m(x, start_pos=0), m(x, start_pos=0))
    check("start_pos=0 is deterministic", eq)

    # Absolute positions cannot wrap; overflow must raise, not alias.
    try:
        m(x[:, :8], start_pos=30)
        check("start_pos past block_size raises", False, "no error raised")
    except ValueError:
        check("start_pos past block_size raises", True)

    # wpe must receive gradient during training.
    m.train()
    m(torch.randint(0, 512, (2, 16)), torch.randint(0, 512, (2, 16))).backward()
    g = m.wpe.weight.grad
    check("wpe receives gradient", g is not None and g.abs().sum() > 0,
          f"|g| = {g.abs().sum():.4f}" if g is not None else "None")


def test_real_config_size():
    """The shipped config must land in the 495M-500M band."""
    from src.model import load_config

    cfg = load_config()
    m = Atom500M(cfg)
    n = m.num_parameters()
    check("shipped config is 495M-500M", 495e6 <= n <= 500e6,
          f"{n:,} ({n / 1e6:.2f}M)")
    check("n_layer is 29", cfg["n_layer"] == 29, str(cfg["n_layer"]))
    check("position embedding is 2,097,152 params",
          m.wpe.weight.numel() == cfg["block_size"] * cfg["d_model"],
          f"{m.wpe.weight.numel():,}")


def test_hf_repo_target():
    """Every HF upload target must point at siddhu1174/atom-500m."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for rel in ["scripts/train_ddp.py", "scripts/run_kaggle.py", "README.md"]:
        text = (root / rel).read_text(encoding="utf-8")
        # Only HF targets; the GitHub clone source legitimately keeps its owner.
        for m_ in re.finditer(r'"([\w.-]+/atom-500m)"', text):
            if m_.group(1) != "siddhu1174/atom-500m" and "github" not in m_.group(1):
                offenders.append(f"{rel}: {m_.group(1)}")
    check("no stale HF repo id", not offenders, "; ".join(offenders))


def test_hf_repo_in_generated_cell():
    """The generated Kaggle cell must target the new repo."""
    import io
    from contextlib import redirect_stdout

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import run_kaggle

    buf = io.StringIO()
    with redirect_stdout(buf):
        cell = run_kaggle.build_cell()
    check("generated cell targets siddhu1174/atom-500m",
          "siddhu1174/atom-500m" in cell)
    check("generated cell still clones from GitHub",
          "github.com/BharathDeva1174/atom-500m" in cell)


def test_sequence_length_guard():
    """Forward must reject sequences longer than block_size."""
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 16, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    try:
        m(torch.randint(0, 512, (1, 17)))
        check("rejects seq_len > block_size", False, "no error raised")
    except AssertionError:
        check("rejects seq_len > block_size", True)


def test_dtype_guard():
    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 16, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    try:
        m(torch.randint(0, 512, (1, 8), dtype=torch.int32))
        check("rejects non-int64 ids", False, "no error raised")
    except AssertionError:
        check("rejects non-int64 ids", True)


def test_dataset_against_real_file(tmp):
    """Round-trip a corpus through .npy and confirm windowing matches memory."""
    tokens = np.arange(5000, dtype=np.int64)
    path = os.path.join(tmp, "tokens.npy")
    np.save(path, tokens)

    ds = PretokenizedDataset(path, block_size=100)
    check("memory-mapped length", len(ds) == (5000 - 1) // 100, str(len(ds)))

    item = ds[10]
    check("x starts at offset", int(item["x"][0]) == 10 * 100, str(int(item["x"][0])))
    check("y is x shifted", torch.equal(item["y"][:-1], item["x"][1:]))

    # The model must consume those already-aligned targets without shifting
    # them again. Shifting twice trains a two-token-ahead predictor: the run
    # scores ~3.9 on that objective, ~8.7 on real next-token prediction, and
    # generates word salad. The datasets already do the shift, so the model's
    # loss has to be the straight-through cross-entropy.
    small_cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
                 "block_size": 100, "ff_expansion": 2.0}
    m = Atom500M(small_cfg).eval()
    in_vocab = PretokenizedDataset((np.arange(1200) % 500).astype(np.int64),
                                   block_size=100)
    pair = in_vocab[3]
    with torch.no_grad():
        logits = m(pair["x"][None])
        straight = torch.nn.functional.cross_entropy(
            logits.reshape(-1, small_cfg["vocab_size"]),
            pair["y"].reshape(-1))
        again = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, small_cfg["vocab_size"]),
            pair["y"][1:].reshape(-1))
        got = m(pair["x"][None], pair["y"])
    check("model loss uses targets as given (no second shift)",
          abs(got.item() - straight.item()) < 1e-5,
          f"{got.item():.4f} vs {straight.item():.4f} "
          f"(double-shift would be {again.item():.4f})")

    # uint16 downcast path
    small = (np.arange(5000) % 1000).astype(np.uint16)
    ds16 = PretokenizedDataset(small, block_size=100)
    check("uint16 corpus loads", len(ds16) == (5000 - 1) // 100)

    # Out-of-range ids must widen rather than silently wrap. 69999 needs int32
    # (it doesn't fit uint16) but int32 is narrower than int64, so int32 is the
    # correct choice here; a value above 2**31 must reach int64.
    mid = np.arange(70000, dtype=np.int64)
    check("ids > 65535 widen past uint16",
          PretokenizedDataset(mid, block_size=100).tokens.dtype == np.int32,
          str(PretokenizedDataset(mid, block_size=100).tokens.dtype))

    huge = np.array([0, 2**33, 5], dtype=np.int64)
    check("ids > 2**31 widen to int64",
          PretokenizedDataset(huge, block_size=2).tokens.dtype == np.int64,
          str(PretokenizedDataset(huge, block_size=2).tokens.dtype))

    # prompt_mask on the tensor path must fail loudly, not guess a boundary.
    try:
        PretokenizedDataset(small, block_size=100, prompt_mask=True)[0]
        check("prompt_mask rejected on token arrays", False, "no error raised")
    except ValueError:
        check("prompt_mask rejected on token arrays", True)


def test_lr_schedule():
    """Warmup then cosine decay; imported from the real trainer."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_ddp as T

    T.WARMUP_STEPS, T.LEARNING_RATE, T.MIN_LR = 10, 1e-3, 1e-5
    lrs = [T.lr_at(s, 100) for s in range(100)]
    check("lr starts near 0 during warmup", lrs[0] < 2e-4, f"{lrs[0]:.2e}")
    check("lr peaks at end of warmup", abs(lrs[9] - 1e-3) < 1e-9, f"{lrs[9]:.2e}")
    check("lr decays monotonically after warmup",
          all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(10, 99)))
    check("lr floors at MIN_LR", abs(lrs[-1] - 1e-5) < 1e-6, f"{lrs[-1]:.2e}")


def test_optimizer_param_groups():
    """LayerNorm and embedding must be excluded from weight decay."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_ddp as T

    m = Atom500M({"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
                  "block_size": 16, "ff_expansion": 3.5})
    opt = T.build_optimizer(m)
    check("two param groups (decay / no_decay)", len(opt.param_groups) == 2)
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    for name, p in m.named_parameters():
        should_decay = p.dim() >= 2 and "wte" not in name
        if should_decay and id(p) not in decayed:
            check(f"{name} decayed", False)
            break
    else:
        check("all 2D non-embedding weights decayed", True)
    check("wte excluded from decay",
          id(m.wte.weight) not in decayed)


def test_checkpoint_barrier_not_guarded():
    """Regression guard for the DDP checkpoint deadlock (FIX 2).

    `_checkpoint` must reach its `all_reduce` on EVERY rank. If the rank-0 guard
    ever moves above the collective again, rank 0 blocks forever waiting for a
    partner that already returned — and the job dies silently at the first
    checkpoint, which on Kaggle means losing the whole session.

    This asserts it structurally: the guard must come *after* the barrier, or
    rather must not enclose it. Rather than re-implement the logic, we check
    that the all_reduce call is outside the `if is_main():` block by inspecting
    the source's AST.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "scripts" / "train_ddp.py").read_text(encoding="utf-8"))

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_checkpoint"),
        None,
    )
    check("_checkpoint exists", fn is not None)
    if fn is None:
        return

    barriers = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_reduce"
    ]
    check("_checkpoint contains an all_reduce barrier", len(barriers) == 1,
          f"found {len(barriers)}")

    # Find the rank-0 guard and confirm the barrier is not nested inside it.
    guards = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Name)
        and n.test.func.id == "is_main"
    ]
    if not guards:
        check("barrier is not nested in a rank-0 guard", True, "no guard at all")
        return

    guard = guards[0]
    nested = [b for b in barriers if b in ast.walk(guard)]
    check(
        "barrier is NOT inside the rank-0 guard",
        not nested,
        "" if not nested else "DEADLOCK: all_reduce nested under if is_main()",
    )


def test_weight_tying_survives_save():
    """Regression guard for the weight-tying serialization bug (FIX 3).

    `state_dict()` returns detach() views, so tied weights share one storage. A
    naive `{k: v.cpu()}` allocates a fresh tensor per key, splitting the tie into
    two copies: +206 MB per checkpoint here, and a model that silently loses
    weight tying when resumed. The dedup must key on storage identity.
    """
    import torch as _t
    from src.sync import _state_dict_to_cpu, build_checkpoint

    cfg = {"vocab_size": 512, "d_model": 64, "n_head": 4, "n_layer": 2,
           "block_size": 32, "ff_expansion": 3.5}
    m = Atom500M(cfg)
    check("model starts tied", m.lm_head.weight is m.wte.weight)

    sd = _state_dict_to_cpu(m.state_dict())
    check("wte/lm_head still one object after .cpu()",
          sd["wte.weight"] is sd["lm_head.weight"])

    ckpt = build_checkpoint(m, None, step=1, config=cfg)
    model_sd = ckpt["model"]
    check("build_checkpoint preserves the tie",
          model_sd["wte.weight"] is model_sd["lm_head.weight"])

    # No extra memory for the duplicate: unique storage must equal model total.
    n_params = sum(p.numel() for p in m.parameters())
    n_unique = sum(t.numel() for t in {id(t): t for t in model_sd.values()}.values())
    check("no duplicated tensor storage", n_params == n_unique,
          f"{n_params:,} params vs {n_unique:,} stored")

    # Round-trip through a real torch.save/load, as the trainer does.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pt")
        _t.save(ckpt, p)
        reloaded = _t.load(p, map_location="cpu", weights_only=False)
    rsd = reloaded["model"]
    check("tie survives save/load round-trip",
          rsd["wte.weight"] is rsd["lm_head.weight"])
    check("wpe survives the round-trip", "wpe.weight" in rsd)

    # And a model can actually be reloaded from it.
    m2 = Atom500M(cfg)
    m2.load_state_dict(reloaded["model"], strict=True)
    check("reloaded model is still tied", m2.lm_head.weight is m2.wte.weight)
    x = _t.randint(0, 512, (1, 8))
    m.eval(); m2.eval()
    with _t.no_grad():
        check("reloaded model reproduces logits",
              _t.allclose(m(x), m2(x), atol=1e-5))


if __name__ == "__main__":
    import tempfile

    print("=" * 66)
    print("Atom-500M integration test (CPU)")
    print("=" * 66)

    print("\n[tiny training loop]")
    m = test_tiny_training_loop()

    print("\n[gradient flow]")
    test_tied_weight_gradients(m)

    print("\n[generation]")
    test_generate()

    print("\n[positional embeddings]")
    test_positional_embeddings()

    print("\n[config sizing]")
    test_real_config_size()

    print("\n[input validation]")
    test_sequence_length_guard()
    test_dtype_guard()

    print("\n[dataset]")
    with tempfile.TemporaryDirectory() as td:
        test_dataset_against_real_file(td)

    print("\n[trainer helpers]")
    test_lr_schedule()
    test_optimizer_param_groups()

    print("\n[HF target]")
    test_hf_repo_target()
    test_hf_repo_in_generated_cell()

    print("\n[checkpoint invariants]")
    test_checkpoint_barrier_not_guarded()
    test_weight_tying_survives_save()

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("All integration checks passed.")
    print("=" * 66)
