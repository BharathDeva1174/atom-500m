"""Streaming, memory-mapped dataset for Atom-500M.

Two input modes, one interface:

  * **Pre-tokenized tensor** — a single flat `uint16`/`int32`/`int64` `.npy` or
    `.pt` file of GPT-2 BPE ids, memory-mapped and concatenated. This is the
    fast path for Kaggle: tokenize once, train many epochs.
  * **Streaming Hugging Face dataset** — tokenized lazily with `tiktoken` if no
    pre-tokenized file exists. Useful for a quick smoke run, but note that
    streaming tokenization *is not reproducible across DDP ranks* (each rank
    walks the shard stream independently and shards are not aligned), so prefer
    the pre-tokenized path for real training.

Each `__getitem__` returns a contiguous window of `block_size + 1` tokens, split
as `x = w[:-1]`, `y = w[1:]` — the standard next-token LM objective. `y` is
returned **unshifted**; `Atom500M.forward` performs the internal shift, so the
pair never drifts out of alignment by one position.
"""

import bisect
import glob
import json
import os
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def _resolve_dtype(arr: np.ndarray) -> np.dtype:
    """Pick the narrowest int dtype that fits this vocab without overflow.

    GPT-2's vocab tops out at 50256, so uint16 is enough and halves the
    memory-mapped footprint. A shard containing a larger id (a different
    tokenizer) is detected here rather than silently wrapping to garbage.
    """
    arr_max = int(arr.max()) if arr.size else 0
    if arr_max < 0:
        raise ValueError("token array contains negative ids")
    if arr_max <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if arr_max <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


class PretokenizedDataset(Dataset):
    """Sliding-window LM dataset over a flat stream of token ids.

    Parameters
    ----------
    data:
        Either a 1-D array/tensor of token ids, or a path to a ``.npy`` / ``.pt``
        file containing one. Also accepts a directory — every ``.npy`` shard in
        it is concatenated in sorted order.
    block_size:
        Tokens per training example (2048 for Atom-500M). The returned window is
        ``block_size + 1`` long.
    hf_dataset:
        Optional ``datasets.Dataset`` (typically ``streaming=True``) to tokenize
        on the fly when no pre-tokenized file is available.
    hf_text_column:
        Name of the text column in ``hf_dataset``.
    num_blocks:
        Cap on the number of windows in the dataset. ``None`` = every valid
        offset. Useful for a fixed-length epoch when streaming from HF.
    prompt_mask:
        If ``True``, and ``hf_dataset`` rows carry ``prompt``/``response``
        columns, mask prompt tokens with ``IGNORE_INDEX`` so loss is computed
        only on the empathetic reply. Requires ``hf_dataset``.
    """

    def __init__(
        self,
        data: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        block_size: int = 2048,
        hf_dataset=None,
        hf_text_column: str = "text",
        num_blocks: Optional[int] = None,
        prompt_mask: bool = False,
        encode_fn=None,
    ):
        self.block_size = block_size
        self.window = block_size + 1
        self.prompt_mask = prompt_mask
        self.hf_text_column = hf_text_column

        if data is not None:
            self.tokens = self._load_tokens(data)
            self._hf = None
            self._prompt_ids = 0
        elif hf_dataset is not None:
            self.tokens = None
            self._hf = hf_dataset
            self._encode = encode_fn or _default_encoder()
            self._buffer = []  # rolling carry-over across streamed rows
            self._buffer_len = 0
            self._prompt_ids = 0
        else:
            raise ValueError("PretokenizedDataset needs either `data` or `hf_dataset`")

        # A flat token array carries no prompt boundary, so masking would have to
        # guess. Reject at construction rather than silently masking an arbitrary
        # prefix of every block, or (worse) never masking at all.
        if prompt_mask and self.tokens is not None:
            raise ValueError(
                "prompt_mask=True requires hf_dataset with prompt/response "
                "columns; a pre-tokenized token array has no boundary "
                "information. Pre-mask the corpus yourself before tokenizing."
            )

        # For the in-memory path, len() is exact. For streaming, it is unknown
        # until the stream is walked, so `num_blocks` is required to size the
        # epoch — otherwise a length_fn would have to consume the whole stream.
        if self.tokens is not None:
            self.num_blocks = len(self) if num_blocks is None else min(num_blocks, len(self))
        elif num_blocks is None:
            raise ValueError(
                "`num_blocks` is required when streaming from Hugging Face: the "
                "stream length is unknown without walking it end to end."
            )
        else:
            self.num_blocks = num_blocks

    # -- loading ------------------------------------------------------------
    def _load_tokens(self, data) -> np.ndarray:
        if isinstance(data, (str, os.PathLike)):
            path = str(data)
            if os.path.isdir(path):
                shards = sorted(
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".npy")
                )
                if not shards:
                    raise FileNotFoundError(f"no .npy shards in {path}")
                print(f"[dataset] concatenating {len(shards)} shard(s) from {path}")
                data = np.concatenate([np.load(s, mmap_mode="r") for s in shards])
            elif path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")  # zero-copy memory map
            elif path.endswith((".pt", ".pth")):
                data = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(data, dict):
                    data = data.get("tokens", next(iter(data.values())))
            else:
                raise ValueError(f"unsupported data file: {path} (expected .npy/.pt)")

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data = np.asarray(data).reshape(-1)

        dtype = _resolve_dtype(data)
        if data.dtype != dtype:
            data = data.astype(dtype)
        print(f"[dataset] {len(data):,} tokens, dtype={np.dtype(dtype).name}")
        return data

    # -- HF streaming -------------------------------------------------------
    def _next_hf_window(self, index: int) -> torch.Tensor:
        """Pull enough rows from the stream to fill window `index`.

        Rows are carried in `self._buffer` across calls so a block can span a row
        boundary. Rows are consumed in stream order, so successive calls walk the
        dataset monotonically — which is why this must be driven by a sequential
        sampler (see `scripts/train_ddp.py`).
        """
        while self._buffer_len < self.window:
            try:
                row = next(self._iter)
            except StopIteration:
                raise IndexError(
                    f"Hugging Face stream exhausted at block {index} "
                    f"({self._buffer_len}/{self.window} tokens buffered)"
                ) from None

            ids = np.asarray(self._encode(row[self.hf_text_column]), dtype=np.int64)
            self._buffer.append(ids)
            self._buffer_len += len(ids)

            # With prompt_mask on, remember how many leading tokens of the
            # *current* window came from a prompt. Recomputed per row; the
            # masking itself is applied in _apply_prompt_mask once the window
            # is full, so this only needs the newest row's prompt length.
            self._prompt_ids = (
                len(self._encode(row["prompt"])) if self.prompt_mask and "prompt" in row else 0
            )

        flat = np.concatenate(self._buffer)
        win = flat[: self.window].copy()
        # Carry the remainder (the row that straddles the block boundary).
        self._buffer = [flat[self.window :]]
        self._buffer_len = len(self._buffer[0])
        return torch.from_numpy(win.astype(np.int64))

    def _ensure_iter(self) -> None:
        if getattr(self, "_iter", None) is None:
            self._iter = iter(self._hf)

    # -- Dataset API --------------------------------------------------------
    def __len__(self) -> int:
        if self.tokens is None:
            return self.num_blocks
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, index: int) -> dict:
        if index >= self.num_blocks:
            raise IndexError(index)

        if self.tokens is not None:
            offset = index * self.block_size
            chunk = np.asarray(self.tokens[offset : offset + self.window], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
            return {"x": x, "y": y}

        self._ensure_iter()
        win = self._next_hf_window(index).long()
        x, y = win[:-1], win[1:]
        return {"x": x, "y": self._apply_prompt_mask(x, y)}

    def _apply_prompt_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Zero out loss on prompt tokens so the model is trained only on the
        empathetic reply, not on reproducing what the user said.

        HF path only — the constructor rejects prompt_mask for token arrays.
        `y[i]` is the loss-bearing target for predicting `x[i+1]`, so a token
        that belongs to the prompt has `y[i]` masked.
        """
        if not self.prompt_mask:
            return y

        n_prompt = len(self._prompt_ids)
        # The window may start mid-prompt or mid-reply; mask the leading
        # min(n_prompt, len(y)) positions that correspond to prompt tokens.
        n_mask = min(n_prompt, y.numel())
        y = y.clone()
        y[:n_mask] = IGNORE_INDEX
        return y

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def describe() -> dict:
        return {
            "objective": "next-token LM; dataset returns y unshifted, model shifts",
            "window": "block_size + 1 contiguous tokens, stride == block_size",
            "dtype": "uint16 when vocab < 65536, else int32/int64",
        }


def _default_encoder():
    """GPT-2 BPE via tiktoken, with a hard failure if it isn't installed."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tiktoken is required to tokenize a Hugging Face dataset on the fly; "
            "pip install tiktoken"
        ) from exc

    enc = tiktoken.get_encoding("gpt2")
    return lambda text: enc.encode(text, allowed_special="set()")


def collate(batch):
    """Stack a list of dataset items into DDP-friendly batch tensors."""
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


class ShardInfo(NamedTuple):
    """One shard's geometry. `n_chunks` is authoritative, not file size — a
    partially-filled trailing shard holds fewer complete chunks."""

    path: str
    n_chunks: int
    total_tokens: int


class ShardedMemmapDataset(Dataset):
    """Fixed-chunk LM dataset over a directory of uint16 `.bin` shards.

    Each shard is a flat run of `block_size + 1`-token chunks written by
    `scripts/prepare_dataset.py`. Unlike `PretokenizedDataset`, which slides an
    overlapping window over one flat array, this reads *disjoint fixed* chunks:
    chunk `i` covers tokens `[i*(B+1), (i+1)*(B+1))`, and `x = chunk[:-1]`,
    `y = chunk[1:]`. The two are equivalent LM objectives; fixed chunks are what
    let a shard be a self-contained file and be memmapped independently.

    **Memory.** Shards are memory-mapped and never fully resident, so a 100 GB
    corpus trains in constant RAM. Only the pages touched by the current batch
    are read, and the OS evicts them under pressure.

    **Worker safety.** `np.memmap` handles are opened lazily and cached *per
    process* under a PID-keyed dict. This is required, not an optimization: a
    DataLoader on Windows uses `spawn`, which pickles the dataset to each worker,
    and an open `np.memmap` cannot be pickled. An eagerly-opened handle would
    raise at `num_workers > 0` on Windows and leak file descriptors on Linux.

    **Shuffling.** A global chunk order is permuted with a seeded RNG, so indices
    are reproducible across ranks (same `seed` -> same permutation) while
    differing across epochs when `epoch` is advanced. Chunk ids are stored, not
    shard objects, so a permutation is a few MB of int64 no matter how large the
    corpus.
    """

    def __init__(
        self,
        shard_paths: Union[str, Sequence[str]],
        block_size: int = 2048,
        dtype: str = "uint16",
        seed: int = 1337,
        shuffle: bool = True,
        epoch: int = 0,
        validate: bool = True,
        vocab_size: Optional[int] = None,
    ):
        self.block_size = block_size
        self.window = block_size + 1
        self.dtype = np.dtype(dtype)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = epoch
        # Per-process memmap cache. Must exist before any _get_mmap call, so it
        # is allocated before validation rather than after.
        self._mmaps: dict = {}

        # Everything below can raise (unreadable shard, out-of-vocab ids, ...),
        # and by then memmaps may already be open. Without this guard a failed
        # construction leaks a handle per shard, and on Windows that turns the
        # real error into an unrelated "file is in use" from whatever cleanup
        # runs next -- the diagnostic you actually need gets buried.
        try:
            self._construct(self._resolve_paths(shard_paths), validate, vocab_size)
        except BaseException:
            self.close()
            raise

    def _construct(self, paths, validate, vocab_size) -> None:
        """Validation, shard geometry, and index setup. Raises on bad input."""
        if not paths:
            raise FileNotFoundError(
                "no .bin shards found. Run scripts/prepare_dataset.py first."
            )

        self.shards: List[ShardInfo] = []
        for p in paths:
            n_tokens = os.path.getsize(p) // self.dtype.itemsize
            n_chunks = n_tokens // self.window
            if n_chunks == 0:
                # A shard smaller than one chunk is a packer bug, not a warning:
                # silently dropping data is how a corpus loses a split unnoticed.
                raise ValueError(
                    f"shard {p} holds {n_tokens} tokens, fewer than one "
                    f"{self.window}-token chunk. Raise --block_size or re-pack."
                )
            if n_tokens % self.window:
                # Trailing partial chunk. The packer truncates these, so this is
                # expected only if a shard was appended to by hand.
                print(f"[shards] {os.path.basename(p)}: dropping "
                      f"{n_tokens % self.window} trailing tokens")
            self.shards.append(ShardInfo(p, n_chunks, n_tokens))

        if validate:
            self._validate_shards(vocab_size=vocab_size)

        self.total_chunks = sum(s.n_chunks for s in self.shards)
        # Offset of each shard's first chunk, for O(log S) index -> shard lookup.
        self._offsets = list(np.cumsum([0] + [s.n_chunks for s in self.shards]))
        self.set_epoch(self.epoch)

    # -- setup --------------------------------------------------------------
    @staticmethod
    def _resolve_paths(shard_paths) -> List[str]:
        if isinstance(shard_paths, (str, os.PathLike)):
            d = str(shard_paths)
            if os.path.isdir(d):
                return sorted(glob.glob(os.path.join(d, "*.bin")))
            return [d]
        return list(shard_paths)

    def _validate_shards(self, sample: int = 1024, vocab_size: Optional[int] = None) -> None:
        """Confirm every shard really is uint16 gpt2 data, cheaply.

        A strided sample across the whole file, not just the endpoints. Reading
        only head and tail is not enough: a float32 shard interpreted as uint16
        yields the low mantissa bits, which at both ends can be small and
        plausible (0 and 17792) while the interior is garbage. Sampling ~1k
        spread-out positions catches that for a few milliseconds of I/O, and a
        corrupt shard would otherwise surface as silently-degraded loss hours
        into training.

        When `vocab_size` is given, the bound is the model's own vocabulary
        rather than a hardcoded gpt2 constant. That turns the most common
        pipeline mistake — training on a corpus tokenized with a different
        tokenizer than the model was configured for — into an error at load
        time instead of an `IndexError` deep inside `F.embedding` on the first
        batch, after the model and optimizer are already on the GPU.
        """
        v_max = vocab_size if vocab_size is not None else 50257
        for s in self.shards:
            arr = self._get_mmap(s.path)
            n = s.total_tokens
            if n == 0:
                continue
            step = max(1, n // sample)
            idx = np.unique(np.concatenate([
                np.array([0, n - 1]), np.arange(0, n, step, dtype=np.int64)
            ]))
            probe = np.asarray(arr[idx], dtype=np.int64)
            bad = probe[(probe < 0) | (probe >= v_max)]
            if bad.size:
                what = (f"corpus contains ids >= {v_max}, but the model is "
                        f"configured for vocab_size={v_max}" if vocab_size else
                        f"does not look like {self.dtype} gpt2 tokens")
                raise ValueError(
                    f"shard {s.path} {what} "
                    f"({bad.size}/{probe.size} sampled ids out of range, e.g. "
                    f"{int(bad[0])}). Was it written with a different dtype, "
                    f"byte order, or tokenizer?"
                )
        self._mmaps.clear()  # release validation handles; reopened on demand

    def _get_mmap(self, path: str) -> np.memmap:
        """Return this process's memmap for `path`, opening it on first use."""
        key = (os.getpid(), path)
        mm = self._mmaps.get(key)
        if mm is None:
            mm = np.memmap(path, dtype=self.dtype, mode="r")
            self._mmaps[key] = mm
        return mm

    def close(self) -> None:
        """Release this process's memmap handles.

        Windows will not unlink a file that is still memory-mapped, and Linux
        leaks a file descriptor per shard per worker — with many shards and
        several workers that adds up. Call this when tearing the dataset down
        (after training, or before deleting a shard directory).
        """
        for mm in self._mmaps.values():
            mmap_obj = getattr(mm, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass  # already closed, or a view is still alive
        self._mmaps.clear()

    def __del__(self):
        # Best-effort: __del__ may run at interpreter shutdown when the
        # internals it touches are already torn down.
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        """Drop memmap handles before pickling to a DataLoader worker.

        `np.memmap` holds an open file object and a live mmap, neither of which
        pickles. Handing workers the paths and geometry instead means each
        process opens its own mapping on first use.
        """
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state

    # -- epoch / shuffle ----------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch, re-deriving the chunk permutation."""
        self.epoch = epoch
        if not self.shuffle:
            self._order = np.arange(self.total_chunks, dtype=np.int64)
            return
        # Seeded on (seed, epoch) so every rank shares one permutation and the
        # order is reproducible across runs, but differs between epochs.
        rng = np.random.default_rng(self.seed + epoch)
        self._order = rng.permutation(self.total_chunks).astype(np.int64)

    # -- Dataset API --------------------------------------------------------
    def __len__(self) -> int:
        return self.total_chunks

    def _locate(self, index: int) -> "tuple[str, int]":
        """Map a permuted index -> (shard path, chunk index within that shard).

        Binary search over cumulative offsets: O(log S), not O(S). With ~1000
        shards and millions of chunks this matters — a linear scan per __getitem__
        would dominate the loader's time.
        """
        chunk_id = int(self._order[index])
        shard_idx = bisect.bisect_right(self._offsets, chunk_id) - 1
        return self.shards[shard_idx].path, chunk_id - int(self._offsets[shard_idx])

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self.total_chunks
        if not (0 <= index < self.total_chunks):
            raise IndexError(f"index {index} out of range for {self.total_chunks} chunks")

        path, chunk_idx = self._locate(index)
        arr = self._get_mmap(path)
        start = chunk_idx * self.window
        # A copy is required: memmap slices are views into a read-only mapping,
        # and torch.from_numpy on such a view warns and can misbehave under
        # pin_memory. 2049 uint16 = ~4 KB, so the copy is trivial.
        chunk = np.array(arr[start : start + self.window], dtype=np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"x": x, "y": y}

    # -- introspection ------------------------------------------------------
    def stats(self) -> dict:
        return {
            "shards": len(self.shards),
            "chunks": self.total_chunks,
            "block_size": self.block_size,
            "tokens": sum(s.total_tokens for s in self.shards),
            "dtype": self.dtype.name,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "epoch": self.epoch,
        }

    @staticmethod
    def describe() -> dict:
        return {
            "layout": "fixed chunks of block_size+1; x=chunk[:-1], y=chunk[1:]",
            "access": "np.memmap per shard, opened lazily per process (spawn-safe)",
            "shuffle": "seeded permutation of chunk ids, re-derived per epoch",
            "dtype": "uint16 (gpt2 vocab < 65536)",
        }


if __name__ == "__main__":
    # Self-test on synthetic data — verifies windowing and the x/y relationship.
    rng = np.random.default_rng(0)
    fake = rng.integers(0, 50257, size=10_000).astype(np.uint16)
    ds = PretokenizedDataset(fake, block_size=128)
    print(json.dumps(ds.describe(), indent=2))
    print(f"len(dataset)          = {len(ds)}   (expect (10000-1)//128 = {(10000 - 1) // 128})")

    item = ds[3]
    x, y = item["x"], item["y"]
    assert x.shape == (128,) and y.shape == (128,)
    # The defining property of the LM objective: y is x shifted left by one.
    assert torch.equal(y[:-1], x[1:]), "y must be x shifted by one position"
    assert int(fake[3 * 128]) == int(x[0])
    print(f"x[0], y[0]            = {int(x[0])}, {int(y[0])}")
    print(f"y[:-1] == x[1:]       = {torch.equal(y[:-1], x[1:])}")
    print(f"x.dtype               = {x.dtype}")

    b = collate([ds[0], ds[1], ds[2]])
    print(f"batched shapes        = {tuple(b['x'].shape)}, {tuple(b['y'].shape)}")
    print("OK — windowing, shift relationship, and collation all clean.")
