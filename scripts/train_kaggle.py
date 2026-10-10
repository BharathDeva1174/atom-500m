"""Kaggle training entry point for Atom-500M on dual T4 GPUs.

Two ways to use it:

**As a Kaggle notebook cell** -- it is importable and self-configuring:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    exec(open("scripts/train_kaggle.py").read())

**As a script** under `torchrun`:

    torchrun --nproc_per_node=2 scripts/train_kaggle.py

Why a separate file from `train_ddp.py`: that one assumes you are already inside
a launched process group and hard-fails without `RANK` in the environment. This
one bootstraps its own group (DDP when >1 GPU, single-process otherwise), which
is what a notebook needs -- a Kaggle cell has no `torchrun` to launch it.

Architecture note
----------------
The model definition is *imported* from `src/model.py`, not redeclared here.
A second `AtomConfig` in this file would be a second source of truth that can
silently disagree with the one the model was built from; `verify_config()` below
asserts the values match the documented spec instead.
"""

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import ShardedMemmapDataset, collate  # noqa: E402
from src.model import Atom500M, load_config  # noqa: E402
from src.sync import (  # noqa: E402
    cleanup_stale_checkpoints,
    save_and_push_checkpoint,
)


# ---------------------------------------------------------------------------
# The documented Atom config. Asserted against configs/atom_500m.json at startup
# so a drift between this spec and the real config is caught in the first second
# rather than by a subtly under-trained model.
# ---------------------------------------------------------------------------
ATOM_CONFIG = {
    "vocab_size": 50257,
    "d_model": 1024,
    "n_layer": 29,
    "n_head": 16,
    "block_size": 2048,
    "ff_expansion": 3.53,
}

# Runtime-only switches, set on the model rather than stored in the JSON config.
# `verify_config` deliberately ignores these: they are training-run decisions
# (is VRAM tight?), not properties of the architecture, and requiring them in
# the config file would make a config change for every memory experiment.
RUNTIME_FLAGS = {"grad_checkpointing"}

HF_REPO = os.environ.get("HF_REPO_ID", "siddhu1174/atom-500m")
SHARD_DIR = Path(os.environ.get("ATOM_SHARD_DIR", "./data/shards"))
CONFIG_PATH = os.environ.get("ATOM_CONFIG", "configs/atom_500m.json")
CHECKPOINT_DIR = os.environ.get("ATOM_CKPT_DIR", "./checkpoints")

# Tuned for T4 x2 (16 GB each). 497M params: fp32 weights 2.0 GB, Adam states
# 4.0 GB, plus fp16 activations. batch 4 x accum 4 keeps peak memory around
# 13 GB while giving a 65,536-token effective batch, which is reasonable for
# 497M. Raise ATOM_GRAD_ACCUM if a step needs to be more stable.
BATCH_SIZE = int(os.environ.get("ATOM_BATCH_SIZE", 4))       # per GPU
GRAD_ACCUM = int(os.environ.get("ATOM_GRAD_ACCUM", 4))
LEARNING_RATE = float(os.environ.get("ATOM_LR", 3e-4))
MIN_LR = float(os.environ.get("ATOM_MIN_LR", 3e-4 * 0.1))
WARMUP_STEPS = int(os.environ.get("ATOM_WARMUP", 200))
WEIGHT_DECAY = float(os.environ.get("ATOM_WD", 0.1))
GRAD_CLIP = 1.0
LOG_EVERY = int(os.environ.get("ATOM_LOG_EVERY", 20))
EVAL_EVERY = int(os.environ.get("ATOM_EVAL_EVERY", 250))
EVAL_BATCHES = int(os.environ.get("ATOM_EVAL_BATCHES", 20))
SAVE_EVERY = int(os.environ.get("ATOM_SAVE_EVERY", 500))
MAX_STEPS = int(os.environ.get("ATOM_MAX_STEPS", 0))        # 0 = one epoch
DOWNLOAD = os.environ.get("ATOM_DOWNLOAD", "1") == "1"
GRAD_CHECKPOINTING = os.environ.get("ATOM_GRAD_CHECKPOINT", "0") == "1"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def verify_config(skip: bool = False) -> dict:
    """Assert the shipped config matches ATOM_CONFIG above.

    Guards against the two drifting apart. `skip` exists for the CPU test
    harness, which deliberately runs a tiny config; it is never set on Kaggle.
    """
    cfg = load_config(CONFIG_PATH)
    if skip:
        return cfg
    for k, want in ATOM_CONFIG.items():
        if k in RUNTIME_FLAGS:
            continue  # training-run switch, not an architecture property
        got = cfg.get(k)
        if got != want:
            raise ValueError(
                f"{CONFIG_PATH} has {k}={got!r}, but scripts/train_kaggle.py "
                f"documents {k}={want!r}. Update whichever is wrong so the "
                f"notebook and the trainer agree."
            )
    log(f"[config] verified: {json.dumps(cfg)}")
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ensure_shards() -> None:
    """Make sure ./data/shards/ holds .bin files, downloading if needed.

    A Kaggle notebook starts with an empty filesystem, so the corpus has to be
    fetched from the Hub. Skipped when shards are already present, so a re-run
    in the same session does not re-download 444 MB.
    """
    existing = sorted(SHARD_DIR.glob("*.bin"))
    if existing:
        log(f"[data] {len(existing)} shard(s) already present in {SHARD_DIR}")
        return

    if not DOWNLOAD:
        raise FileNotFoundError(
            f"no shards in {SHARD_DIR} and ATOM_DOWNLOAD=0. Attach a Kaggle "
            f"dataset containing data/shards/*.bin, or set ATOM_DOWNLOAD=1."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required to download shards") from exc

    if not HF_TOKEN:
        raise SystemExit(
            f"No shards in {SHARD_DIR} and no HF_TOKEN set.\n"
            f"On Kaggle, add HF_TOKEN as a notebook secret first:\n"
            f"  from kaggle_secrets import UserSecretsClient\n"
            f"  os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')"
        )

    log(f"[data] downloading data/ from {HF_REPO} ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset" if os.environ.get("HF_REPO_TYPE") == "dataset" else "model",
        allow_patterns=["data/*.bin"],
        token=HF_TOKEN or None,
        local_dir=".",
    )
    n = len(sorted(SHARD_DIR.glob("*.bin")))
    if not n:
        raise SystemExit(
            f"download completed but no .bin files landed in {SHARD_DIR}. "
            f"Check that {HF_REPO} has a data/ folder containing shards."
        )
    log(f"[data] downloaded {n} shard(s)")


def build_loaders(config: dict, rank: int, world_size: int):
    """Train/eval loaders over the sharded corpus.

    Both are memory-mapped, so peak RAM is independent of corpus size. Eval
    uses a distinct seed and `shuffle=False` so the loss curve is measured
    against a stable sample rather than whatever the epoch permutation happens
    to produce.
    """
    ensure_shards()

    train_ds = ShardedMemmapDataset(
        SHARD_DIR, block_size=config["block_size"], seed=1337,
        vocab_size=config["vocab_size"],
    )
    eval_ds = ShardedMemmapDataset(
        SHARD_DIR, block_size=config["block_size"], seed=1338, shuffle=False,
        vocab_size=config["vocab_size"],
    )
    log(f"[data] {json.dumps(train_ds.stats())}")

    train_sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=1337
    )
    eval_sampler = torch.utils.data.DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    # num_workers=2 per rank: on a 2-GPU notebook that is 4 loader processes
    # competing with 2 trainer processes for CPU, which is the usual Kaggle
    # bottleneck. The shards are memmapped, so workers cost little memory.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate, drop_last=True, num_workers=2, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, sampler=eval_sampler,
        collate_fn=collate, drop_last=True, num_workers=0, pin_memory=True,
    )
    return train_loader, eval_loader, train_sampler, train_ds


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
def setup() -> tuple:
    """Bootstrap a process group, or run single-process if that is all there is.

    Returns (rank, local_rank, world_size, device, amp_context, scaler).

    `ATOM_FORCE_CPU=1` swaps CUDA for gloo on CPU. That is not a Kaggle mode --
    it exists so the training loop can be tested on a machine without a GPU,
    which is the only way to catch accumulation, clipping, and barrier bugs
    before spending a 12-hour session on them.
    """
    force_cpu = os.environ.get("ATOM_FORCE_CPU", "0") == "1"
    n_gpu = torch.cuda.device_count()

    if force_cpu:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device("cpu")
        # DDP requires a process group even for a single process, so this is
        # initialized unconditionally. Skipping it at world_size==1 makes every
        # single-process run fail at the DDP constructor instead of training.
        if dist.is_available():
            init = os.environ.get("ATOM_INIT_METHOD")
            if init:
                dist.init_process_group(
                    backend="gloo", init_method=init, rank=rank,
                    world_size=world_size,
                )
            else:
                dist.init_process_group(
                    backend="gloo", rank=rank, world_size=world_size,
                )

        def amp():
            return contextlib.nullcontext()

        return rank, 0, world_size, device, amp, torch.amp.GradScaler("cuda", enabled=False)

    if not torch.cuda.is_available() or n_gpu == 0:
        raise SystemExit(
            "No CUDA GPU visible. On Kaggle, enable the GPU accelerator "
            "(Settings -> Accelerator -> GPU T4 x2) and re-run."
        )

    launched = "RANK" in os.environ
    if launched:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    elif n_gpu > 1:
        # torchrun was not used (the usual notebook case). Launch it ourselves
        # rather than silently training on one GPU: the user asked for dual T4.
        log(f"[setup] no torchrun detected, launching {n_gpu} processes")
        return relaunch_via_torchrun(n_gpu)
    else:
        rank, local_rank, world_size = 0, 0, 1
        torch.cuda.set_device(0)

    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    log(f"[setup] rank {rank}/{world_size} on {torch.cuda.get_device_name(local_rank)}")

    def amp():
        return torch.cuda.amp.autocast(dtype=torch.float16)

    scaler = torch.amp.GradScaler("cuda")
    return rank, local_rank, world_size, device, amp, scaler


def relaunch_via_torchrun(n_gpu: int):
    """Re-exec under torchrun, then exit this process.

    `os.execv` replaces the current process image, so the caller must treat the
    return as terminal. In a notebook this matters: without it the cell would
    train on one GPU while believing it had two.
    """
    import subprocess
    import sys as _sys

    cmd = [
        _sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpu}", "--master_port",
        os.environ.get("MASTER_PORT", "29500"),
        os.path.abspath(__file__),
    ]
    log(f"[setup] exec: {' '.join(cmd)}")
    sys.stdout.flush()
    os.execv(_sys.executable, cmd)
    raise SystemExit("unreachable")  # execv does not return


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def lr_at(step: int, total: int) -> float:
    """Linear warmup, then cosine decay to MIN_LR."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only.

    LayerNorm gains and biases, and the tied embedding, are excluded: decaying
    gains shrinks them toward zero every step, which measurably hurts
    convergence. `wte` is excluded explicitly -- it is 51M parameters, and
    weight decay on a tied embedding is doubly wrong because the same tensor is
    also the output classifier.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if (p.dim() >= 2 and "wte" not in name) else no_decay).append(p)

    log(f"[optim] {len(decay)} decayed, {len(no_decay)} undecayed")
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8,
    )


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches: int) -> float:
    """Mean eval loss, summed across ranks then divided once.

    Averaging per-rank means would over-weight whichever rank ran fewer batches;
    summing (loss_sum, count) and dividing at the end is the true mean.
    """
    model.eval()
    loss_sum = n_batches = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp():
            loss_sum += model(x, y).item()
        n_batches += 1

    stats = torch.tensor([loss_sum, n_batches], dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    return float(stats[0] / stats[1].clamp(min=1.0))


def checkpoint(model, optimizer, step, config) -> None:
    """Rank-0 saves and uploads, then every rank crosses the barrier.

    The work is rank-0-only but the `all_reduce` is not: an earlier version
    early-returned on non-zero ranks, so rank 0 blocked forever on a partner
    that had already left, and the job died silently at the first checkpoint.
    On a 12-hour Kaggle session that means losing everything.
    """
    path = None
    if is_main():
        if not HF_TOKEN:
            print("[ckpt] no HF_TOKEN — saving locally only")
        t0 = time.time()
        path = save_and_push_checkpoint(
            model, optimizer, step, repo_id=HF_REPO,
            token=HF_TOKEN or None, config=config,
            checkpoint_dir=CHECKPOINT_DIR, keep_local=not HF_TOKEN,
        )
        if path:
            log(f"[ckpt] step {step} saved in {time.time() - t0:.1f}s -> {path}")

    status = torch.tensor([0 if path else 1], dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) == 1:
        log("[ckpt] WARNING: checkpoint failed on rank 0 — this step is not recoverable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rank, local_rank, world_size, device, amp, scaler = setup()
    on_gpu = device.type == "cuda"
    # The CPU harness runs a deliberately tiny config, so the spec assertion is
    # skipped there. On Kaggle (and under torchrun) it always runs.
    config = verify_config(skip=(not on_gpu and os.environ.get(
        "ATOM_FORCE_CPU", "0") == "1"))

    model = Atom500M(config).to(device)
    n_params = model.num_parameters()
    log(f"[model] {n_params:,} parameters ({n_params / 1e6:.2f}M)")
    if not (495e6 <= n_params <= 500e6):
        log(f"[model] WARNING: outside the 495M-500M target band")

    # Gradient checkpointing trades ~30% step time for a large cut in
    # activation VRAM. Only meaningful in training mode; the model bypasses it
    # in eval, so eval cost is unchanged.
    model.grad_checkpointing = GRAD_CHECKPOINTING
    if GRAD_CHECKPOINTING:
        log("[model] gradient checkpointing ON (activations recomputed in backward)")

    ddp_model = DDP(
        model,
        device_ids=[local_rank] if on_gpu else None,
        broadcast_buffers=False,
    )
    optimizer = build_optimizer(ddp_model)

    train_loader, eval_loader, sampler, train_ds = build_loaders(
        config, rank, world_size
    )
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = MAX_STEPS or steps_per_epoch
    if is_main():
        cleanup_stale_checkpoints(CHECKPOINT_DIR, keep=1)

    eff_tokens = BATCH_SIZE * GRAD_ACCUM * world_size * config["block_size"]
    log(f"[train] {steps_per_epoch} steps/epoch | total {total_steps:,} | "
        f"effective batch ~{eff_tokens:,} tokens")

    step = 0
    try:
        for epoch in range(10_000 if not MAX_STEPS else 1):
            sampler.set_epoch(epoch)
            ddp_model.train()
            running = t0 = 0.0
            optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                with amp():
                    loss = ddp_model(x, y)
                scaler.scale(loss / GRAD_ACCUM).backward()
                running += loss.item()

                if (it + 1) % GRAD_ACCUM:
                    continue

                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step, total_steps)
                # unscale -> clip -> step. Clipping scaled gradients is a no-op,
                # and the scaler must be allowed to skip inf/nan steps itself.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % LOG_EVERY == 0:
                    if on_gpu:
                        torch.cuda.synchronize()
                    dt = max(1e-9, time.time() - t0)
                    avg = running / (LOG_EVERY * GRAD_ACCUM)
                    mem = (f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                           if on_gpu else "cpu")
                    # Throughput is meaningless on the CPU harness (it is a
                    # correctness test, not a benchmark), and rounding to 0
                    # tok/s reads like a bug rather than a non-measurement.
                    rate = (f"{LOG_EVERY * eff_tokens / dt:,.0f} tok/s"
                            if on_gpu else "n/a")
                    log(f"step {step}/{total_steps:,} | loss {avg:.4f} | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                        f"{rate} | {mem}")
                    running = t0 = 0.0

                if step % EVAL_EVERY == 0:
                    log(f"step {step} | eval {evaluate(ddp_model, eval_loader, device, amp, EVAL_BATCHES):.4f}")

                if step % SAVE_EVERY == 0:
                    checkpoint(ddp_model, optimizer, step, config)

                if MAX_STEPS and step >= MAX_STEPS:
                    break
            if MAX_STEPS and step >= MAX_STEPS:
                break

        checkpoint(ddp_model, optimizer, step, config)
        log(f"[done] {step:,} steps complete")

    except KeyboardInterrupt:
        log("[interrupt] checkpointing before exit")
        checkpoint(ddp_model, optimizer, step, config)
        raise
    except Exception as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        checkpoint(ddp_model, optimizer, step, config)
        raise
    finally:
        train_ds.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        log("[cleanup] process group destroyed")


if __name__ == "__main__":
    main()
