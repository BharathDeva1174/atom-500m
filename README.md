# Atom-500M

A decoder-only GPT built for one thing: high-EQ conversational empathy. No code,
no math, no technical domain logic.

**Status:** architecture, data pipeline, and distributed trainer are implemented
and tested. **No training corpus ships with this repo** — see [Data](#data).

---

## Read this first

The model is **497.49M parameters** with learned absolute positional embeddings.
`python src/model.py` prints the measured count and a full component breakdown on
every run, and asserts the 495M–500M band.

One consequence to know before you build a chat interface: **learned absolute
positions cap the sequence at `block_size` total tokens.** A 1900-token prompt
can generate 148 more tokens, not unbounded. That is the standing trade-off of
absolute embeddings, and the reason RoPE and ALiBi exist. `model.generate()`
checks the budget up front and fails with an actionable message instead of dying
mid-generation. See `NOTES.md` §2.

`NOTES.md` documents every deviation from the original brief and why.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Does the model build, and is the parameter count what you expect?
python src/model.py

# 2. Print the Kaggle notebook cell — paste it into a T4 x2 session.
python scripts/run_kaggle.py

# 3. Verify everything (38 checks, CPU, ~2 min).
python scripts/integration_test.py
```

## Layout

```
configs/atom_500m.json     hyperparameters
src/model.py               Atom500M — pre-LN decoder, SwiGLU, tied embeddings
src/dataset.py             ShardedMemmapDataset (fixed chunks, memmap) + PretokenizedDataset
src/sync.py                checkpoint save → Hugging Face upload → free disk
scripts/prepare_dataset.py tokenize + pack dialogues into uint16 .bin shards
scripts/train_ddp.py       DDP training engine (assumes torchrun)
scripts/train_kaggle.py    self-bootstrapping trainer for Kaggle notebooks
scripts/run_kaggle.py      generates the ready-to-paste Kaggle cell
scripts/integration_test.py    50 CPU checks of model, data, and trainer
scripts/test_shard_pipeline.py multi-shard, multi-worker, and edge cases
scripts/test_train_kaggle.py   config guard, schedule, loader wiring
scripts/test_kaggle_loop.py    real 2-rank training loop on CPU/gloo
scripts/ddp_smoke_test.py       real 2-rank DDP run on CPU/gloo
```

## Architecture

Pre-LayerNorm decoder-only transformer, GPT-2 lineage.

| | |
|---|---|
| Layers | 29 |
| `d_model` | 1024 |
| Heads | 16 (head_dim 64) |
| Context | 2048 |
| FFN | SwiGLU, hidden 3616 (3.53 × `d_model`) |
| Attention | `F.scaled_dot_product_attention(is_causal=True)` → FlashAttention on CUDA |
| Embeddings | **tied** — `lm_head.weight` *is* `wte.weight` |
| Positional | learned absolute, 2048 × 1024 |

Measured breakdown (from `python src/model.py`):

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding (tied) | 51,463,168 | 10.3% |
| Position embedding | 2,097,152 | 0.4% |
| Attention (29 layers) | 121,634,816 | 24.4% |
| SwiGLU (29 layers) | 322,142,208 | 64.8% |
| LayerNorms + biases | 150,528 | 0.03% |
| **Total** | **497,487,872** | |

## Data

Shards are built by `scripts/prepare_dataset.py` from raw dialogue JSONL or a
Hugging Face dataset, then memory-mapped at training time.

```bash
# The full four-dataset empathy mixture (~233M tokens, 5 shards)
python scripts/prepare_dataset.py --mixture

# One dataset
python scripts/prepare_dataset.py --dataset_name facebook/empathetic_dialogues \
    --trust_remote_code

# Pack and upload
python scripts/prepare_dataset.py --mixture \
    --push_to_hf --repo_id siddhu1174/atom-500m
```

The mixture and what each reader actually does — these are four different
schemas, none of which is a plain text column:

| Dataset | Schema | Handling | Yields |
|---|---|---|---|
| `facebook/empathetic_dialogues` | one utterance per row; `prompt` recurs per `conv_id` | grouped into whole conversations | 17.8k convs |
| `allenai/soda` | `dialogue` + `speakers` lists, multi-party | first speaker is Person, rest Listener | 9.1M turns |
| `OpenAssistant/oasst2` | a **tree** (`parent_id`/`message_id`), multilingual | one path per root, first-child at each fork; non-`en` dropped | 5.1k paths |
| `LuangMV97/Empathetic_counseling_Dataset` | flat `input`/`label` pairs | one Person/Listener pair per row | 30.9k pairs |

Two of these are easy to get silently wrong. **empathetic_dialogues** repeats its
context on every row, so a naive per-row reader yields 76k near-duplicate
fragments instead of 17.8k conversations. **oasst2** is a tree, and a node with
four children means linearizing the rows produces text where the assistant
appears to answer a question it never saw — so only single paths are emitted.

Every document is terminated with `<|endoftext|>` (50256) so chunk boundaries land
on dialogue boundaries, and the stream is cut into fixed 2049-token chunks —
one chunk per example, `x = chunk[:-1]`, `y = chunk[1:]`. Shards are written to
`data/shards/shard_NNNN.bin` as flat `uint16`, capped at ~100 MB each.

**Text is cleaned before tokenizing.** Dataset punctuation artifacts
(`_comma_`, `_blank_`, `_period_`, `_question_`, `_exclamation_`) become real
punctuation — left in, the model learns to emit the literal escape word.
Typographic quotes and dashes fold to ASCII, zero-width characters become
spaces, and whitespace runs collapse. Accents are preserved.

**Why fixed chunks rather than a sliding window.** Both are valid LM objectives,
but fixed chunks let each shard be a self-contained file of exactly N chunks,
which is what makes independent memmapping and 100 MB sharding possible. A
sliding window would make every chunk overlap its neighbours and waste 99.9% of
the corpus on redundant context.

**Memory is independent of corpus size.** Tokens stream from the tokenizer
straight to disk; only one shard buffer (~100 MB) is ever resident. A flat
`.npy` from a prior workflow still works via `ATOM_DATA` as a fallback.

The trainer validates each shard on load — a strided sample across the whole
file, bounded by the model's own `vocab_size`. A corpus tokenized with a
different tokenizer fails at load time rather than deep inside `F.embedding` on
the first batch.

### Record formats

`--format auto` (the default) handles turn lists (`conversation`/`messages`),
prompt/response pairs, and plain text columns, mapping `human`/`gpt` speakers to
`Person`/`Listener` — speaker labels carry real signal for an empathy model.
Override with `--format conversation|pairs|plain`. The four mixture datasets use
their own readers regardless of `--format`.

No corpus ships with this repo. For a model whose entire purpose is empathy,
*which* dialogue you train on — and its consent and licensing posture — is a
decision worth making deliberately.

## Training

On Kaggle, run the generated cell — it downloads the corpus, clones the repo,
and starts both T4s:

```bash
python scripts/run_kaggle.py     # prints the notebook cell
```

`scripts/train_kaggle.py` is the notebook entry point. Unlike `train_ddp.py`, it
bootstraps its own process group, so a plain `python scripts/train_kaggle.py`
inside a notebook cell launches two workers itself rather than silently training
on one GPU. Under `torchrun` it uses the existing group as-is.

| Variable | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token. Without it, checkpoints stay local. |
| `HF_REPO_ID` | `siddhu1174/atom-500m` | Corpus download + checkpoint upload target. |
| `ATOM_SHARD_DIR` | `./data/shards` | Auto-downloaded from the Hub if empty. |
| `ATOM_BATCH_SIZE` | 4 | Per GPU. |
| `ATOM_GRAD_ACCUM` | 4 | 4 × 4 × 2 GPUs × 2048 = **65,536 tokens/step**. |
| `ATOM_LR` | 3e-4 | Cosine-decayed to 3e-5, 200-step warmup. |
| `ATOM_SAVE_EVERY` | 500 | Steps between checkpoints. |
| `ATOM_MAX_STEPS` | 0 | 0 = run one full epoch. |
| `ATOM_FORCE_CPU` | 0 | CPU/gloo path, for testing the loop without a GPU. |

Mixed precision is fp16 autocast with a GradScaler — T4 is Turing-class, so bf16
is unavailable. Gradients clip at 1.0, after `unscale_()` and before
`scaler.step()`, which is the only ordering where clipping means anything.

`train_kaggle.py` asserts at startup that `configs/atom_500m.json` still matches
the architecture it documents. Two copies of a config is two places to be wrong;
this makes a drift fail in the first second rather than as a subtly under-trained
model.

### Checkpoints

Save locally → upload to the Hub → **delete the local copy**. Kaggle gives ~20 GB
and kills sessions without warning, so disk stays flat regardless of run length.
Uploads are best-effort: a failure logs and continues rather than killing a run
that is hours in. `NOTES.md` §5 covers the two subtleties (weight-tying
preservation, non-persistent buffers).

## Testing

```bash
python scripts/integration_test.py     # 50 checks, CPU, ~2 min
python scripts/test_shard_pipeline.py  # shard rotation, workers, edge cases
python scripts/test_train_kaggle.py    # config guard, schedule, loader wiring
python scripts/test_kaggle_loop.py     # real 2-rank training loop, CPU/gloo
python scripts/prepare_dataset.py --self_test
python scripts/ddp_smoke_test.py       # real 2-rank DDP run, ~1 min
```

The suites cover the training loop converging, gradient flow, position
sensitivity, the 495M–500M size band, dataset windowing, the LR schedule,
optimizer param groups, and the HF repo target. Two checks are **mutation-tested**
— I verified the deadlock guard actually fails when a broken `_checkpoint` is
injected, since a guard that cannot fail is worse than none.

The shard suite covers the two paths a single-shard test cannot reach: chunk
boundaries straddling shard files (verified across 65 shards), and DataLoader
workers, which on Windows `spawn` and pickle the dataset — the reason memmap
handles are opened lazily per process.

The DDP smoke test packs its corpus with the *real* `prepare_dataset.py` and
trains on 11 shards, so it exercises pack → memmap → DDP together rather than a
hand-written fixture. It provisions everything it needs from a fresh clone.

**Not covered by local tests:** FlashAttention dispatch, NCCL all-reduce,
GradScaler behavior under real fp16 overflow, and the Hub upload path. Those
need actual NVIDIA hardware.

## Requirements

Python 3.10+, PyTorch 2.2+. Developed and tested against torch 2.7.1.
