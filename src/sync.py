"""Checkpoint offload: save locally, push to Hugging Face Hub, delete locally.

Kaggle notebooks get ~20 GB of scratch disk and are killed without warning. A
500M model in fp32 is ~2 GB per checkpoint; Adam states add ~4 GB more. So the
policy is: write one checkpoint, upload it, delete the local copy, move on. Disk
usage stays flat at roughly one checkpoint regardless of how long training runs.

Every failure mode here is non-fatal. A network blip, an expired token, or a
full disk must never take down a training run that is hours in.
"""

import os
import shutil
import time
from typing import Optional

import torch

CHECKPOINT_DIR = "./checkpoints"

# A bare torch.save of a 416M fp32 model is ~1.7 GB. Leave headroom above that.
_MIN_FREE_BYTES_BEFORE_SAVE = 3 * 1024**3


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from a DDP / DataParallel / compiled wrapper.

    DDP's own `module` attribute is the standard path; the loop covers nested
    wrappers (e.g. DDP(compile(model))) and `compile()`'s `_orig_mod`.
    """
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile wrapper
        model = model._orig_mod
    return model


def _free_bytes(path: str) -> int:
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
    return usage.free


def _state_dict_to_cpu(state_dict: dict) -> dict:
    """Move a state dict to CPU while preserving weight-tying identity.

    `state_dict()` returns `param.detach()` views — distinct objects sharing one
    storage for tied weights. A plain `{k: v.cpu()}` would allocate a fresh
    tensor per key, splitting the tie into two copies (+206 MB per checkpoint
    here, and a model that silently loses weight tying when resumed). So
    tensors are deduped by storage pointer: the first key pointing at a storage
    owns the CPU copy, and every later key with the same storage reuses it.
    """
    out: dict = {}
    by_storage: dict = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        # Untyped storage data_ptr is the identity that survives .detach()/.cpu().
        key = v.untyped_storage().data_ptr() if v.numel() else None
        if key is not None and key in by_storage:
            out[k] = by_storage[key]  # same tensor object — the tie survives
            continue
        cpu_tensor = v.detach().cpu()
        out[k] = cpu_tensor
        if key is not None:
            by_storage[key] = cpu_tensor
    return out


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    config: Optional[dict] = None,
    loss: Optional[float] = None,
) -> dict:
    """Assemble the checkpoint dict. Exposed separately so tests can inspect
    the contents without touching the filesystem.

    `loss` is the most recent training loss, recorded so a resumed run can be
    sanity-checked against the run that produced it.
    """
    module = _unwrap(model)
    return {
        "step": step,
        "model": _state_dict_to_cpu(module.state_dict()),
        "optimizer": (
            {k: v for k, v in optimizer.state_dict().items()} if optimizer else None
        ),
        "config": config,
        "loss": loss,
        "saved_at": time.time(),
    }


def save_local_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    config: Optional[dict] = None,
    loss: Optional[float] = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> Optional[str]:
    """Write a checkpoint to disk and return its path, or None on failure.

    Saving is deliberately *synchronous* while the upload is not. `torch.save`
    reads tensors that the training loop is concurrently mutating -- the
    optimizer state in particular -- so doing it from a background thread risks
    writing a torn checkpoint: a file that loads without error but whose
    optimizer state is half from step N and half from step N+1. Serializing on
    the main thread makes the snapshot consistent; only the network wait, which
    is the slow and variable part, moves off-thread.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    try:
        torch.save(build_checkpoint(model, optimizer, step, config, loss), path)
    except Exception as exc:
        print(f"[sync] ERROR: local save failed at step {step}: {exc}")
        return None
    return path


def save_and_push_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    repo_id: str,
    token: Optional[str] = None,
    config: Optional[dict] = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
    keep_local: bool = False,
) -> Optional[str]:
    """Save a checkpoint and push it to the Hugging Face Hub.

    Returns the remote path on success, or None if the push failed. Never
    raises: a failed checkpoint is a lost optimization step, not a lost run.
    The local file is removed only after the upload is confirmed.

    Parameters
    ----------
    keep_local:
        Set True to skip the local delete (for local debugging, or to keep a
        resume point). Off by default because Kaggle disk is the constraint.
    """
    local_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    remote_path = f"checkpoints/step_{step}.pt"

    # --- 1. save locally ---------------------------------------------------
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        free = _free_bytes(checkpoint_dir)
        if free < _MIN_FREE_BYTES_BEFORE_SAVE:
            print(
                f"[sync] WARNING: only {free / 1024**3:.1f} GB free — checkpoint "
                f"may fail to write"
            )
        torch.save(build_checkpoint(model, optimizer, step, config), local_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"[sync] saved {local_path} ({size_mb:.0f} MB)")
    except Exception as exc:
        print(f"[sync] ERROR: local save failed at step {step}: {exc}")
        return None

    # --- 2. push to the Hub ----------------------------------------------
    # No token means no authenticated upload is possible. Check *before* calling
    # the Hub: an unauthenticated request still hits the network, burns ~seconds
    # on a retry ladder, and returns a 401 for a repo that may well exist.
    if not token:
        print(f"[sync] no token supplied — skipping upload, checkpoint is local only")
        return f"local:{local_path}"

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            commit_message=f"checkpoint step_{step}",
        )
        print(f"[sync] uploaded step_{step}.pt -> {repo_id}/{remote_path}")
    except Exception as exc:
        # Deliberately swallowed. The local file is kept so the checkpoint is
        # not lost — the next successful push (or a manual retry) recovers it.
        print(f"[sync] ERROR: upload failed at step {step}: {exc}")
        print(f"[sync] checkpoint retained locally at {local_path}")
        return None

    # --- 3. free the disk --------------------------------------------------
    if keep_local:
        print(f"[sync] keeping local checkpoint (keep_local=True)")
    else:
        try:
            os.remove(local_path)
            print(f"[sync] deleted local checkpoint, disk reclaimed")
        except Exception as exc:
            print(f"[sync] WARNING: could not delete {local_path}: {exc}")

    return f"{repo_id}/{remote_path}"


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    """Restore a checkpoint and return the step it was saved at.

    Handles the two weight-tying subtleties that bite on resume:
      * `lm_head.weight` is tied to `wte.weight`, so loading the raw state dict
        populates the same tensor twice — harmless, but noted here.
      * `causal_mask` is a non-persistent buffer and is simply absent from the
        saved dict, so `strict=True` is still correct.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    module = _unwrap(model)
    module.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[sync] resumed from {path} at step {ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0))


def cleanup_stale_checkpoints(checkpoint_dir: str = CHECKPOINT_DIR, keep: int = 1) -> None:
    """Remove all but the `keep` most recent local checkpoints.

    Insurance against a crash mid-upload leaving 2 GB orphans behind. Runs on
    rank 0 only.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    steps = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step_") and name.endswith(".pt"):
            try:
                steps.append((int(name[5:-3]), os.path.join(checkpoint_dir, name)))
            except ValueError:
                continue
    steps.sort()
    for _, path in steps[:-keep] if keep > 0 else steps:
        try:
            os.remove(path)
            print(f"[sync] removed stale checkpoint {path}")
        except Exception as exc:
            print(f"[sync] WARNING: could not remove {path}: {exc}")


if __name__ == "__main__":
    from model import Atom500M, load_config

    cfg = load_config()
    m = Atom500M(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    # Exercise unwrapping through a fake DDP-shaped wrapper.
    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    ckpt = build_checkpoint(FakeDDP(FakeDDP(m)), opt, step=7, config=cfg)
    print(f"checkpoint keys   : {sorted(ckpt.keys())}")
    print(f"step              : {ckpt['step']}")
    print(f"model tensors     : {len(ckpt['model'])}")
    print(f"wte.weight shape  : {tuple(ckpt['model']['wte.weight'].shape)}")
    print(f"lm_head tied      : {ckpt['model']['wte.weight'] is ckpt['model']['lm_head.weight']}")
    print(f"causal_mask saved : {'blocks.0.attn.causal_mask' in ckpt['model']} (non-persistent buffer)")
    print(f"optimizer groups  : {len(ckpt['optimizer']['param_groups'])}")

    n_params = sum(p.numel() for p in m.parameters())
    n_unique = sum(t.numel() for t in {id(t): t for t in ckpt["model"].values()}.values())
    print(f"params            : {n_params:,} model / {n_unique:,} unique in ckpt")
    print(f"untied overhead   : {n_params - n_unique:,} params "
          f"({(n_params - n_unique) * 4 / 1024**2:.0f} MB would be wasted)")

    assert ckpt["model"]["wte.weight"] is ckpt["model"]["lm_head.weight"], "weights must stay tied"
    assert "blocks.0.attn.causal_mask" not in ckpt["model"]
    assert n_params == n_unique, "every tied weight must be stored exactly once"
    print("OK — unwrap, tie preservation, and non-persistent buffer all clean.")
    print("      (no upload attempted: set HF_REPO_ID/HF_TOKEN to test for real)")
