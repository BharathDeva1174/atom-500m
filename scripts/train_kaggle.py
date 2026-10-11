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
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

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
    download_latest_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_and_push_checkpoint,
    save_local_checkpoint,
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
# Cadences are deliberately decoupled.
#
# EVAL_EVERY (100) is frequent: an evaluation writes no files and costs only a
# bounded number of forward passes, so a fine-grained loss curve is nearly free.
# It is the main signal for noticing divergence early -- with only a 500-step
# view, a run that has already gone wrong gives no clue for hours.
#
# SAVE_EVERY (500) is 5x coarser because a checkpoint is ~5.6 GiB: writing and
# uploading one is the only real I/O in the loop, and at 100 it saturates the
# connection for the length of a session.
#
# Decoupling costs nothing in eval freshness here, because 500 is a multiple of
# 100: every checkpoint step is also an eval step, so the cached loss a
# checkpoint records was measured at that exact step. Verified on disk --
# step_500.pt carries eval_step=500, gap 0.
#
# That is a property of *these* two numbers, not a guarantee. Setting
# SAVE_EVERY=250 would leave each checkpoint holding the eval from step 200, a
# 50-step gap. `eval_step` is recorded for exactly that reason: it is what
# distinguishes a fresh measurement from a carried-over one.
EVAL_EVERY = int(os.environ.get("ATOM_EVAL_EVERY", 100))
EVAL_BATCHES = int(os.environ.get("ATOM_EVAL_BATCHES", 20))
# Every 500 steps. A 497M checkpoint is ~5.6 GiB (fp32 weights 1.85 + AdamW
# exp_avg 1.85 + exp_avg_sq 1.85, i.e. 12 bytes/param), so this is a real
# data-loss/speed trade, not a cosmetic interval. At 100 the upload rate is 5x
# higher and saturates a residential connection; 500 costs at most 500 steps of
# progress on a hard kill. See KEEP_LOCAL_COUNT for the disk math.
SAVE_EVERY = int(os.environ.get("ATOM_SAVE_EVERY", 500))
MAX_STEPS = int(os.environ.get("ATOM_MAX_STEPS", 0))        # 0 = one epoch
DOWNLOAD = os.environ.get("ATOM_DOWNLOAD", "1") == "1"
GRAD_CHECKPOINTING = os.environ.get("ATOM_GRAD_CHECKPOINT", "0") == "1"
# Background upload: how many checkpoints may be queued unsent, and how long
# shutdown waits for the queue to drain. The bound caps disk use if the network
# is slower than training; the timeout stops a hung upload from blocking exit.
UPLOAD_QUEUE = int(os.environ.get("ATOM_UPLOAD_QUEUE", 3))
UPLOAD_JOIN_TIMEOUT = float(os.environ.get("ATOM_UPLOAD_JOIN_TIMEOUT", 300.0))
# How many confirmed checkpoints to retain locally. Measured at ~5.6 GiB each,
# so 2 is ~11.2 GiB -- comfortable on Kaggle's ~20 GB alongside a 0.43 GiB
# corpus, but only while uploads keep pace.
#
# This is the single deletion policy, and it deliberately supersedes the older
# ATOM_KEEP_LOCAL / delete-after-upload pair. Those were independent settings,
# which was a bug: with delete-after-upload on (the default) the uploader
# removed each file the moment its upload finished, so the retention count had
# nothing left to retain and keep-2 silently did nothing. Retention by count is
# strictly more expressive -- 1 gives the old "delete after upload" behavior --
# so one setting replaces two.
#
# Deletion only ever touches checkpoints confirmed on the Hub. A file still
# uploading is protected regardless of age, because it is the only local copy:
# if that upload then fails, pruning it would leave the run unresumable.
KEEP_LOCAL_COUNT = int(os.environ.get("ATOM_KEEP_LOCAL_COUNT", 2))

# Backwards-compatible alias. ATOM_KEEP_LOCAL=1 meant "keep every checkpoint",
# which is ATOM_KEEP_LOCAL_COUNT=0 (retain all); =0 meant the old default of
# delete-immediately, which is a count of 1. Without this, a user carrying
# ATOM_KEEP_LOCAL=1 from an earlier version would silently lose all retention.
_legacy_keep = os.environ.get("ATOM_KEEP_LOCAL")
if _legacy_keep is not None and "ATOM_KEEP_LOCAL_COUNT" not in os.environ:
    KEEP_LOCAL_COUNT = 0 if _legacy_keep == "1" else 1

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

    if not any(SHARD_DIR.glob("*.bin")):
        # The corpus was uploaded with --path_in_repo data, so the Hub layout
        # is data/shard_NNNN.bin and snapshot_download preserves it under
        # local_dir -- one level above ATOM_SHARD_DIR (data/shards). The 444 MB
        # is already on disk; moving it down beats failing the run.
        stray = sorted(SHARD_DIR.parent.glob("*.bin"))
        if stray:
            for p in stray:
                p.rename(SHARD_DIR / p.name)
            log(f"[data] moved {len(stray)} shard(s) from "
                f"{SHARD_DIR.parent}/ into {SHARD_DIR}/")

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

    The reduction tensor MUST live on the training device. NCCL — the backend
    for a CUDA run — cannot reduce CPU tensors and raises
    "No backend type associated with device type cpu". Gloo accepts CPU
    tensors, which is why this only ever shows up on GPU. Following the
    training device works under both backends; hardcoding CPU works under gloo
    and crashes under nccl.
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

    stats = torch.tensor([loss_sum, n_batches], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    return float(stats[0].item() / stats[1].clamp(min=1.0).item())


class AsyncUploader:
    """Upload checkpoints from a background thread so training never stalls.

    The training loop calls `submit()`, which enqueues a path and returns
    immediately. A single daemon thread drains the queue, so a slow or failing
    upload cannot block a step -- the thing that matters on a 12-hour Kaggle
    session with a ~2 GB file on a residential connection.

    **Backpressure.** The queue is bounded. If uploads fall behind (poor
    network), a growing queue would fill the disk with unsent checkpoints and
    eventually kill the run. When the bound is hit, the *oldest* pending upload
    is dropped and its file deleted: recent checkpoints are the valuable ones,
    and the model is still moving forward.

    **Shutdown.** `close()` drains the queue before returning, so a checkpoint
    submitted just before exit actually gets uploaded. The main thread joins
    with a timeout, because a hung upload must not hang the process forever --
    that is how a completed run turns into a lost session.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        path_in_repo: str = "checkpoints",
        max_pending: int = 3,
    ):
        self.repo_id = repo_id
        self.token = token
        self.path_in_repo = path_in_repo
        self.max_pending = max_pending
        # Note there is no delete-after-upload here. Deleting on upload success
        # and pruning by retention count were two policies fighting over the same
        # files, and the faster one always won -- leaving the retention count a
        # no-op. The uploader now only marks a file confirmed; cleanup decides
        # what actually gets removed, and it is driven by KEEP_LOCAL_COUNT.

        self._queue: "queue.Queue" = queue.Queue(maxsize=max_pending)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self.uploaded = 0
        self.failed = 0
        self.dropped = 0
        # Paths confirmed to exist on the Hub. Read by the retention step so a
        # checkpoint still in flight is never pruned -- deleting the only local
        # copy of an unuploaded checkpoint is how a run ends up unresumable.
        self._confirmed: set = set()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker, name="ckpt-uploader", daemon=True
        )
        self._thread.start()
        print(f"[upload] background uploader -> {self.repo_id}/{self.path_in_repo}/")

    def close(self, timeout: float = 300.0) -> None:
        """Drain the queue, join the thread, and report."""
        self._stop.set()
        if self._thread is not None:
            # Join with a bounded timeout: an in-flight upload can hang, and a
            # finished training run must not be held hostage by the network.
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"[upload] WARNING: uploader still running after "
                      f"{timeout:.0f}s; exiting with uploads in flight")
            self._thread = None
        with self._lock:
            print(f"[upload] finished: {self.uploaded} uploaded, "
                  f"{self.failed} failed, {self.dropped} dropped")

    # -- producer -----------------------------------------------------------
    def submit(self, path: str) -> None:
        """Enqueue a checkpoint. Returns immediately, never raises."""
        if path is None or self._thread is None:
            return
        try:
            self._queue.put_nowait(path)
        except queue.Full:
            # Drop the oldest: recent checkpoints matter more than stale ones.
            try:
                old = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                old = None
            if old:
                self.dropped += 1
                print(f"[upload] queue full; dropping {os.path.basename(old)}")
                _quiet_remove(old)
            try:
                self._queue.put_nowait(path)
            except queue.Full:
                self.dropped += 1
                _quiet_remove(path)

    def pending(self) -> int:
        return self._queue.qsize()

    def confirmed(self) -> set:
        """Absolute paths known to be on the Hub."""
        with self._lock:
            return set(self._confirmed)

    def in_flight(self, checkpoint_dir: str) -> set:
        """Local checkpoints that exist but are not yet confirmed on the Hub.

        These must survive retention: they are the only copy of that step.
        """
        confirmed = self.confirmed()
        out = set()
        if not os.path.isdir(checkpoint_dir):
            return out
        for name in os.listdir(checkpoint_dir):
            if name.startswith("step_") and name.endswith(".pt"):
                p = os.path.abspath(os.path.join(checkpoint_dir, name))
                if p not in confirmed:
                    out.add(p)
        return out

    # -- consumer -----------------------------------------------------------
    def _worker(self) -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
        except Exception as exc:
            print(f"[upload] disabled: {type(exc).__name__}: {exc}")
            self._stopped.set()
            return

        while True:
            try:
                # A short timeout keeps the thread responsive to close() instead
                # of blocking on an empty queue.
                path = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue

            try:
                name = os.path.basename(path)
                remote = f"{self.path_in_repo}/{name}"
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=remote,
                    repo_id=self.repo_id,
                    commit_message=f"checkpoint {name}",
                )
                with self._lock:
                    self.uploaded += 1
                    self._confirmed.add(os.path.abspath(path))
                print(f"[upload] {name} -> {remote} ({time.time() - t0:.1f}s)")
                # Not deleted here -- cleanup_stale_checkpoints applies
                # KEEP_LOCAL_COUNT. Deleting at upload time would leave nothing
                # to retain and would discard the local fallback before the
                # next checkpoint exists.
            except Exception as exc:
                # A failed upload must not kill the thread: the run should
                # continue and keep checkpointing.
                with self._lock:
                    self.failed += 1
                print(f"[upload] FAILED {os.path.basename(path)}: "
                      f"{type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()


def _quiet_remove(path: str) -> None:
    """Delete a file, ignoring failure. Windows can hold locks transiently."""
    try:
        os.remove(path)
    except OSError:
        pass


def maybe_resume() -> Optional[str]:
    """Locate a checkpoint to resume from, or None to start fresh.

    Order: an explicit ATOM_RESUME path, then a local checkpoint, then the Hub.
    Local is preferred over remote because on Kaggle the two are usually the
    same file, and reading local avoids a multi-GB download.

    A corrupt or unreadable local checkpoint falls through to the next source
    rather than aborting: on a restarted notebook the file on disk is more
    likely to be a partial write than a real resume point, and refusing to
    start would waste the whole session.
    """
    explicit = os.environ.get("ATOM_RESUME", "")
    if explicit:
        if os.path.exists(explicit):
            return explicit
        print(f"[resume] ATOM_RESUME={explicit} not found; searching instead")

    local = find_latest_checkpoint(CHECKPOINT_DIR)
    if local:
        return local

    if os.environ.get("ATOM_NO_DOWNLOAD_RESUME", "0") == "1":
        return None

    try:
        return download_latest_checkpoint(
            HF_REPO, token=HF_TOKEN or None, checkpoint_dir=CHECKPOINT_DIR
        )
    except Exception as exc:
        # A resume failure must never be fatal: training from scratch beats
        # not training at all.
        print(f"[resume] remote lookup failed ({type(exc).__name__}: {exc}); "
              f"starting from step 0")
        return None


# Last step written by `checkpoint`, so a shutdown save at the same step as the
# scheduled one is a no-op rather than a duplicate ~2 GB upload. A one-element
# list rather than a global, so it resets cleanly per process.
_LAST_SAVED_STEP = [-1]


def checkpoint(model, optimizer, step, config, loss=None, eval_loss=None,
               eval_step=None, uploader=None, device=None) -> None:
    """Save on rank 0, enqueue for upload, then every rank crosses the barrier.

    The work is rank-0-only but the `all_reduce` is not: an earlier version
    early-returned on non-zero ranks, so rank 0 blocked forever on a partner
    that had already left, and the job died silently at the first checkpoint.
    On a 12-hour Kaggle session that means losing everything.

    The save stays on this thread (see `save_local_checkpoint` for why) and
    only the upload is handed to the background thread.

    Idempotent per step. Training that ends exactly on a SAVE_EVERY boundary
    hits this twice -- once from the schedule, once from shutdown -- which would
    write the same ~2 GB file again and push a duplicate commit to the Hub. The
    step number is the natural key for "already saved".
    """
    already_saved = _LAST_SAVED_STEP[0] == step

    if is_main() and not already_saved:
        t0 = time.time()
        path = save_local_checkpoint(
            model, optimizer, step, config=config, loss=loss,
            eval_loss=eval_loss, eval_step=eval_step,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        ok = path is not None
        if path:
            size_mb = os.path.getsize(path) / 1024**2
            log(f"[ckpt] step {step} saved ({size_mb:.0f} MB) "
                f"in {time.time() - t0:.1f}s")
            if uploader is not None:
                uploader.submit(path)
                # Retain the last N *confirmed* checkpoints. In-flight files are
                # protected: at 5.6 GiB each, pruning one that has not reached
                # the Hub yet would leave the run with no resume point.
                cleanup_stale_checkpoints(
                    CHECKPOINT_DIR, keep=KEEP_LOCAL_COUNT,
                    protect=uploader.in_flight(CHECKPOINT_DIR),
                )
            elif not HF_TOKEN:
                # No uploader and no token: the local copy is the only one, so
                # it must not be deleted.
                print("[ckpt] no uploader and no HF_TOKEN — kept locally")
            else:
                # Synchronous fallback: correctness over speed.
                save_and_push_checkpoint(
                    model, optimizer, step, repo_id=HF_REPO, token=HF_TOKEN,
                    config=config, checkpoint_dir=CHECKPOINT_DIR,
                )
    else:
        ok = already_saved

    # Every rank reaches the barrier exactly once, including on a skipped save.
    # The status tensor follows the training device: nccl cannot reduce CPU
    # tensors, so a hardcoded CPU tensor crashes every checkpoint on GPU.
    # Keeping the collective on the accelerator also makes it a device sync
    # point, so the enqueue below cannot race the next step's kernels.
    status = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) == 1:
        log("[ckpt] WARNING: local save failed on rank 0 — "
            "this step is not recoverable")
    if is_main() and ok:
        _LAST_SAVED_STEP[0] = step


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
    # Trust the same retention budget used during the run. Startup cleanup at a
    # different count would prune the extra resume point the previous session
    # deliberately kept -- e.g. keep=1 here would delete the step-0 checkpoint
    # on startup, before any new save existed to replace it.
    if is_main():
        cleanup_stale_checkpoints(CHECKPOINT_DIR, keep=KEEP_LOCAL_COUNT)

    eff_tokens = BATCH_SIZE * GRAD_ACCUM * world_size * config["block_size"]
    log(f"[train] {steps_per_epoch} steps/epoch | total {total_steps:,} | "
        f"effective batch ~{eff_tokens:,} tokens")

    # The uploader only ever runs on rank 0. Starting it on every rank would
    # have each rank uploading the same file.
    uploader = None
    if is_main() and HF_TOKEN:
        uploader = AsyncUploader(
            repo_id=HF_REPO, token=HF_TOKEN, path_in_repo="checkpoints",
            max_pending=UPLOAD_QUEUE,
        )
        uploader.start()
    elif is_main():
        log("[ckpt] no HF_TOKEN — checkpoints stay local (no background upload)")

    # --- resume -------------------------------------------------------------
    # Every rank must agree on the starting step: they share a sampler, and a
    # disagreement would feed different data to different GPUs. The decision is
    # made from the filesystem, which is identical on every rank in a Kaggle
    # session, so no collective is needed to agree.
    step = 0
    start_epoch = 0
    if is_main():
        path = maybe_resume()
    else:
        path = None
    if dist.is_initialized():
        # Broadcast rank 0's decision so a rank that missed the file still agrees.
        flag = torch.tensor(
            [1 if path else 0], dtype=torch.int32, device=device
        )
        dist.broadcast(flag, src=0)
        if not path and int(flag.item()) == 1:
            # Rank 0 found a checkpoint; every rank must wait for it to land.
            dist.barrier()
            path = find_latest_checkpoint(CHECKPOINT_DIR)

    if path:
        try:
            start_step = load_checkpoint(path, ddp_model, optimizer)
        except RuntimeError as exc:
            # Architecture drift between the checkpoint and the current config
            # (e.g. wpe 1024x1024 vs 2048x1024 after a block_size change) makes
            # the file unusable. Same policy as a failed remote lookup in
            # maybe_resume(): training from scratch beats not training at all.
            # Every rank loads the same file, so they all take this branch and
            # stay in agreement on step 0.
            log(f"[resume] {os.path.basename(path)} is incompatible with the "
                f"current config: {str(exc).splitlines()[0][:140]}")
            log("[resume] starting from step 0")
            start_step = 0
        step = start_step
        start_epoch = start_step // max(1, steps_per_epoch)
        _LAST_SAVED_STEP[0] = start_step
        log(f"[resume] continuing from step {start_step} (epoch {start_epoch})")
    else:
        log("[resume] no prior checkpoint — starting from step 0")

    # Running metrics cached so a checkpoint can carry the most recent values
    # without re-running an evaluation. `latest_eval_step` records the step the
    # eval loss was actually measured at -- with decoupled cadences it is
    # usually not the checkpoint's own step.
    latest_train_loss = None
    latest_eval_loss = None
    latest_eval_step = None
    try:
        # Resume lands mid-epoch when steps_per_epoch does not divide the saved
        # step. `skip_micro` counts the batches already consumed inside the
        # current epoch so they are not trained on a second time; the sampler is
        # seeded by epoch, so restarting at the same epoch index reproduces the
        # same batch order and the skip lands on exactly the right batch.
        skip_micro = step * GRAD_ACCUM % max(1, steps_per_epoch * GRAD_ACCUM)

        for epoch in range(start_epoch, 10_000 if not MAX_STEPS else start_epoch + 1):
            sampler.set_epoch(epoch)
            ddp_model.train()
            running = t0 = 0.0
            optimizer.zero_grad(set_to_none=True)

            start_it = skip_micro if epoch == start_epoch else 0
            if start_it:
                log(f"[resume] epoch {epoch}: skipping {start_it} consumed batches")

            for it, batch in enumerate(train_loader):
                if it < start_it:
                    continue
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                with amp():
                    loss = ddp_model(x, y)
                scaler.scale(loss / GRAD_ACCUM).backward()
                running += loss.item()
                latest_train_loss = loss.item()

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
                    latest_eval_loss = evaluate(ddp_model, eval_loader, device,
                                                amp, EVAL_BATCHES)
                    latest_eval_step = step
                    log(f"step {step} | eval {latest_eval_loss:.4f}")

                if step % SAVE_EVERY == 0:
                    # Cadences are decoupled, so this carries the most recent
                    # evaluation, not one from this step. `latest_eval_step`
                    # records where it actually came from so a reader can never
                    # mistake it for a measurement at the checkpoint's step.
                    checkpoint(ddp_model, optimizer, step, config,
                               loss=latest_train_loss,
                               eval_loss=latest_eval_loss,
                               eval_step=latest_eval_step,
                               uploader=uploader, device=device)

                if MAX_STEPS and step >= MAX_STEPS:
                    break
            if MAX_STEPS and step >= MAX_STEPS:
                break

        checkpoint(ddp_model, optimizer, step, config,
                   loss=latest_train_loss,
                   eval_loss=latest_eval_loss,
                   eval_step=latest_eval_step,
                   uploader=uploader, device=device)
        log(f"[done] {step:,} steps complete")

    except KeyboardInterrupt:
        log("[interrupt] checkpointing before exit")
        checkpoint(ddp_model, optimizer, step, config,
                   loss=latest_train_loss,
                   eval_loss=latest_eval_loss,
                   eval_step=latest_eval_step,
                   uploader=uploader, device=device)
        raise
    except Exception as exc:
        log(f"[error] {type(exc).__name__}: {exc}")
        checkpoint(ddp_model, optimizer, step, config,
                   loss=latest_train_loss,
                   eval_loss=latest_eval_loss,
                   eval_step=latest_eval_step,
                   uploader=uploader, device=device)
        raise
    finally:
        # Drain the upload queue BEFORE tearing anything down. A daemon thread is
        # killed at interpreter exit, so a checkpoint submitted moments earlier
        # would never reach the Hub -- and the run would be "successful" with
        # its final weights missing.
        if uploader is not None:
            uploader.close(timeout=UPLOAD_JOIN_TIMEOUT)
        train_ds.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        log("[cleanup] process group destroyed")


if __name__ == "__main__":
    main()
