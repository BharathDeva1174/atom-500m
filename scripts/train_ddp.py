"""DDP multi-GPU training engine for Atom-500M.

Launch:
    torchrun --nproc_per_node=2 scripts/train_ddp.py

Designed for the Kaggle T4 x2 notebook session. Key choices:

  * **fp16 autocast + GradScaler.** T4 is Turing-class: fp16 Tensor Cores are
    ~2x fp32 throughput and 16 GB fits 416M params + Adam states + activations
    at this width. GradScaler is mandatory — fp16 gradients underflow without it.
    Note the ceiling: plain fp16 autocast (not bf16) is used because bf16 needs
    Ampere+, and T4 predates it.

  * **Gradient clipping at 1.0.** Standard for conversational SFT, where a batch
    of an unusually strong empathetic reply can spike the loss.

  * **Zero-redundancy optimizer states.** This script keeps it simple: each rank
    holds its own Adam state and relies on DDP gradient all-reduce. At 416M
    params that is ~5 GB of optimizer state per GPU — fine for a 16 GB T4.
    ZeRO would be needed past ~1B params.

  * **Rank-0 checkpointing.** Only rank 0 saves and uploads, then broadcasts a
    failure flag so no rank proceeds on a stale resume path.
"""

import contextlib
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# torchrun launches this file directly; make `src/` importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (  # noqa: E402
    PretokenizedDataset,
    ShardedMemmapDataset,
    collate,
)
from src.model import Atom500M, load_config  # noqa: E402
from src.sync import (  # noqa: E402
    cleanup_stale_checkpoints,
    load_checkpoint,
    save_and_push_checkpoint,
)

# ---------------------------------------------------------------------------
# Hyperparameters. Env-overridable so the Kaggle cell can tune without an edit.
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("ATOM_CONFIG", "configs/atom_500m.json")
# Shard directory is the primary corpus format; DATA_PATH is the flat-.npy
# fallback kept for the original single-file workflow.
SHARD_DIR = os.environ.get("ATOM_SHARD_DIR", "data/shards")
DATA_PATH = os.environ.get("ATOM_DATA", "data/tokens.npy")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

BATCH_SIZE = int(os.environ.get("ATOM_BATCH_SIZE", 8))  # per GPU
GRAD_ACCUM = int(os.environ.get("ATOM_GRAD_ACCUM", 2))  # effective 16 with 2 GPUs
LEARNING_RATE = float(os.environ.get("ATOM_LR", 3e-4))
MIN_LR = float(os.environ.get("ATOM_MIN_LR", 3e-5))
WARMUP_STEPS = int(os.environ.get("ATOM_WARMUP", 200))
WEIGHT_DECAY = float(os.environ.get("ATOM_WD", 0.1))
GRAD_CLIP = 1.0
SAVE_EVERY = int(os.environ.get("ATOM_SAVE_EVERY", 500))
EVAL_EVERY = int(os.environ.get("ATOM_EVAL_EVERY", 100))
LOG_EVERY = int(os.environ.get("ATOM_LOG_EVERY", 20))
MAX_EPOCHS = int(os.environ.get("ATOM_MAX_EPOCHS", 1))
COMPILE = os.environ.get("ATOM_COMPILE", "0") == "1"

HF_FALLBACK_REPO = "siddhu1174/atom-500m"


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:
    """Rank-0-only logging, line-buffered so Kaggle's cell output stays live."""
    if is_main():
        print(msg, flush=True)


# NCCL is required for the real T4 run, but it does not exist on CPU-only hosts.
# Falling back to gloo keeps the trainer runnable (and testable) off-GPU instead
# of failing with a cryptic "no backend" error.
#
# device_count(), not is_available(): a host can have the CUDA runtime and driver
# libraries installed with zero visible GPUs, and is_available() returns True
# there — which would send us down the CUDA path and then fail on set_device.
# ATOM_FORCE_CPU=1 overrides for local testing.
_FORCE_CPU = os.environ.get("ATOM_FORCE_CPU", "0") == "1"
USE_CUDA = (not _FORCE_CPU) and torch.cuda.device_count() > 0
BACKEND = os.environ.get("ATOM_BACKEND", "nccl" if USE_CUDA else "gloo")

# torchrun's default rendezvous is env:// -> TCPStore, which needs libuv. Some
# torch builds (notably 2.7.x on Windows) ship without it, so ATOM_INIT_METHOD
# allows a file:// store for local multi-process testing. Leave unset on Kaggle.
INIT_METHOD = os.environ.get("ATOM_INIT_METHOD") or None


def amp():
    """fp16 autocast on CUDA; a no-op context on CPU (no fp16 CPU kernels).

    The T4 run uses fp16 because it is Turing-class: bf16 needs Ampere+. On CPU
    the GradScaler is disabled instead, which keeps the numerics identical
    without half-precision kernels.
    """
    if USE_CUDA:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler():
    # torch.amp.GradScaler is the non-deprecated spelling; the device arg is
    # required there but unused when disabled.
    return torch.amp.GradScaler("cuda", enabled=USE_CUDA)


def setup_distributed():
    """Initialize the process group and bind this rank to its device."""
    if "RANK" not in os.environ:
        raise RuntimeError(
            "LOCAL_RANK/RANK not found — launch with:\n"
            "  torchrun --nproc_per_node=2 scripts/train_ddp.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if USE_CUDA:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        where = f"cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})"
    else:
        device = torch.device("cpu")
        where = "cpu (no CUDA visible — gloo fallback)"

    if INIT_METHOD:
        # file:// rendezvous does not read RANK/WORLD_SIZE from the environment;
        # they must be passed explicitly or it raises "rank parameter missing".
        dist.init_process_group(
            backend=BACKEND, init_method=INIT_METHOD, rank=rank, world_size=world_size
        )
    else:
        dist.init_process_group(backend=BACKEND)
    log(f"[setup] rank {rank}/{world_size} on {where}, backend={BACKEND}")
    return rank, local_rank, world_size, device


def lr_at(step: int, total: int) -> float:
    """Linear warmup then cosine decay to MIN_LR."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    progress = min(1.0, progress)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only.

    LayerNorm biases, LayerNorm gains, and the tied embedding are excluded —
    decaying gains shrinks them toward zero every step, which is well known to
    hurt convergence.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "wte" not in name:
            decay.append(p)
        else:
            no_decay.append(p)

    log(f"[optim] {len(decay)} decayed tensors, {len(no_decay)} undecayed")
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),   # GPT-2 beta2; 0.95 is more stable for small LRs
        eps=1e-8,
    )


def build_loaders(block_size: int, world_size: int, rank: int):
    """Build train/eval loaders.

    Uses a DistributedSampler so each rank sees a disjoint shard of the corpus —
    without it, both T4s compute identical gradients and the 2x is wasted.
    """
    vocab_size = None
    if os.path.isfile(CONFIG_PATH):
        vocab_size = load_config(CONFIG_PATH).get("vocab_size")
    if os.path.isdir(SHARD_DIR):
        # Preferred path: uint16 .bin shards from scripts/prepare_dataset.py.
        # Memory-mapped, so corpus size does not bound RAM. vocab_size comes from
        # the model config so a tokenizer mismatch fails here, not in F.embedding.
        train_ds = ShardedMemmapDataset(SHARD_DIR, block_size=block_size, seed=1337,
                                        vocab_size=vocab_size)
        eval_ds = ShardedMemmapDataset(SHARD_DIR, block_size=block_size,
                                       seed=1338, shuffle=False,
                                       vocab_size=vocab_size)
        log(f"[data] shards: {json.dumps(train_ds.stats())}")
    elif os.path.exists(DATA_PATH):
        train_ds = PretokenizedDataset(DATA_PATH, block_size=block_size)
        eval_ds = PretokenizedDataset(DATA_PATH, block_size=block_size)
    else:
        raise FileNotFoundError(
            f"No training data. Expected either:\n"
            f"  - a shard directory at {SHARD_DIR} "
            f"(python scripts/prepare_dataset.py --input ...)\n"
            f"  - a flat token array at {DATA_PATH}"
        )

    train_sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=1337
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate, drop_last=True, num_workers=2, pin_memory=True,
    )
    eval_sampler = torch.utils.data.DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, sampler=eval_sampler,
        collate_fn=collate, drop_last=True, num_workers=0, pin_memory=True,
    )
    log(f"[data] {len(train_ds)} train blocks, {len(eval_loader)} eval batches "
        f"({BATCH_SIZE} x {GRAD_ACCUM} x {world_size} = "
        f"{BATCH_SIZE * GRAD_ACCUM * world_size} effective batch)")
    return train_loader, eval_loader, train_sampler


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> float:
    """Mean loss over a capped number of eval batches, reduced across ranks.

    The eval loader is a DistributedSampler shard, so each rank sees a different
    slice. Summing (total, count) via all_reduce and dividing once gives the true
    mean over the union — averaging per-rank means would over-weight whichever
    rank happened to run fewer batches.
    """
    model.eval()
    # Accumulate as a plain Python float, then build the tensor once at the end.
    # Adding a 2-element tensor in-place to a preallocated one trips torch's
    # broadcast rules, and a float16 sum over many batches would lose precision.
    loss_sum, n_batches = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp():
            loss = model(x, y)
        loss_sum += loss.item()
        n_batches += 1

    # SUM, so every rank ends up with the mean over all shards.
    stats = torch.tensor([loss_sum, float(n_batches)], dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    model.train()
    return float(stats[0] / stats[1].clamp(min=1.0))


def main():
    rank, local_rank, world_size, device = setup_distributed()

    config = load_config(CONFIG_PATH)
    log(f"[config] {json.dumps(config)}")

    model = Atom500M(config).to(device)
    log(f"[model] {model.num_parameters():,} parameters "
        f"({model.num_parameters() / 1e6:.1f}M)")

    if COMPILE:
        # ~2x step time after a ~1 min warmup. Off by default: on a 12h Kaggle
        # session with frequent checkpoints, the compile cost rarely pays back.
        model = torch.compile(model)

    ddp_model = DDP(
        model,
        device_ids=[local_rank] if USE_CUDA else None,
        broadcast_buffers=False,
    )

    optimizer = build_optimizer(ddp_model)
    scaler = make_scaler()

    # Loaders first: total step count depends on the dataset length, and resume
    # needs that count to map a saved step back to an epoch.
    train_loader, eval_loader, train_sampler = build_loaders(config["block_size"], world_size, rank)

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * MAX_EPOCHS

    # --- resume ------------------------------------------------------------
    start_step, start_epoch = 0, 0
    resume_path = os.environ.get("ATOM_RESUME", "")
    if resume_path and os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, ddp_model, optimizer)
        start_epoch = start_step // max(1, steps_per_epoch)
        log(f"[resume] continuing from step {start_step}, epoch {start_epoch}")

    if start_step == 0 and is_main():
        cleanup_stale_checkpoints(keep=1)

    log(f"[train] {steps_per_epoch} optimizer steps/epoch, {total_steps} total, "
        f"warmup {WARMUP_STEPS}")

    # --- training loop -----------------------------------------------------
    global_step = start_step
    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            train_sampler.set_epoch(epoch)
            model.train()
            running, t0 = 0.0, time.time()

            optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                with amp():
                    loss = ddp_model(x, y)
                scaler.scale(loss / GRAD_ACCUM).backward()

                running += loss.item()

                # Step only at the accumulation boundary.
                if (it + 1) % GRAD_ACCUM != 0:
                    continue

                lr = lr_at(global_step, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                scaler.unscale_(optimizer)      # must unscale before clipping
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)           # scaler skips the step on inf/nan
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    if USE_CUDA:
                        torch.cuda.synchronize()
                    avg = running / (LOG_EVERY * GRAD_ACCUM)
                    tokens = LOG_EVERY * GRAD_ACCUM * BATCH_SIZE * config["block_size"] * world_size
                    log(f"step {global_step}/{total_steps} | loss {avg:.4f} | "
                        f"lr {lr:.2e} | {tokens / (time.time() - t0):,.0f} tok/s")
                    running, t0 = 0.0, time.time()

                if global_step % EVAL_EVERY == 0:
                    eval_loss = evaluate(ddp_model, eval_loader, device)
                    log(f"step {global_step} | eval loss {eval_loss:.4f}")

                if global_step % SAVE_EVERY == 0:
                    _checkpoint(ddp_model, optimizer, global_step, config)

            # End-of-epoch checkpoint: the notebook session can die between
            # scheduled saves, so the epoch boundary is a guaranteed sync point.
            _checkpoint(ddp_model, optimizer, global_step, config)

        log(f"[done] training complete at step {global_step}")

    except KeyboardInterrupt:
        log("[interrupt] caught — checkpointing before exit")
        _checkpoint(ddp_model, optimizer, global_step, config)
        raise
    except Exception as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        _checkpoint(ddp_model, optimizer, global_step, config)
        raise
    finally:
        # Always tear down the process group, or NCCL leaves the GPUs wedged
        # and the next torchrun in the same notebook hangs.
        if dist.is_initialized():
            dist.destroy_process_group()
        log("[cleanup] process group destroyed")


def _checkpoint(model, optimizer, step, config) -> None:
    """Checkpoint + upload, with the failure broadcast to all ranks.

    The save/upload *work* is rank-0-only, but the collective is NOT. Every rank
    must reach the all_reduce below: if rank 0 blocks there waiting for a partner
    that early-returned, the job deadlocks until the session is killed. That is
    why the `is_main()` guard wraps only the work, never the barrier.
    """
    path = None
    if is_main():
        repo_id = HF_REPO_ID or HF_FALLBACK_REPO
        if not HF_TOKEN:
            print(f"[ckpt] no HF_TOKEN set — saving locally only "
                  f"(not uploading to {repo_id})")

        t0 = time.time()
        path = save_and_push_checkpoint(
            model, optimizer, step, repo_id=repo_id,
            token=HF_TOKEN or None, config=config,
            # Without a token there is no upload to make room for, so the local
            # copy is the only one — it must not be deleted.
            keep_local=not HF_TOKEN,
        )
        if path:
            log(f"[ckpt] step {step} saved in {time.time() - t0:.1f}s")

    # Barrier on CPU so it works under both the nccl and gloo backends.
    # MIN means "if any participating rank failed, everyone learns it".
    status = torch.tensor([0 if path else 1], dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)

    if int(status.item()) == 1:
        log("[ckpt] WARNING: checkpoint failed on rank 0 — training continues "
            "but this step is NOT recoverable")


if __name__ == "__main__":
    main()
