# Atom-500M — Engineering Notes

Deviations from the original spec, and why. Everything below is a deliberate
decision; nothing was changed silently.

---

## 1. Parameter count: 497.49M

The spec's config was internally inconsistent — it produced 416.5M, not the
495–500M the name implies. The model now ships at **497,487,872 parameters**,
inside the target band, with `src/model.py` asserting the range on every run.

```json
{ "vocab_size": 50257, "d_model": 1024, "n_layer": 29,
  "n_head": 16, "block_size": 2048, "ff_expansion": 3.53 }
```

| Component | Count | Share |
|---|---:|---:|
| Token embedding (tied with `lm_head`) | 51,463,168 | 10.3% |
| Position embedding | 2,097,152 | 0.4% |
| Attention, 29 layers | 121,634,816 | 24.4% |
| SwiGLU, 29 layers | 322,142,208 | 64.8% |
| LayerNorms + biases | 150,528 | 0.03% |
| **Total** | **497,487,872** | |

**Why `n_layer: 29` and not the spec's 24.** Depth is the cheapest axis to grow at
fixed width — it costs activation memory, not optimizer memory. Going 24 → 29
adds ~76M.

**Why `ff_expansion: 3.53` and not the spec's 3.5.** A layer is 15.27M
parameters, so the layer count alone is too coarse to land inside a 5M band:

| Config | Result |
|---|---:|
| `n_layer: 29, ff: 3.5` | 494.64M — just *under* the band |
| **`n_layer: 29, ff: 3.53`** | **497.49M — inside** |
| `n_layer: 29, ff: 3.55` | 500.34M — just *over* |
| `n_layer: 30, ff: 3.5` | 509.85M |

Nudging the FFN width to land the total precisely also required relaxing the
hidden-width rounding from 128 to 32; hidden 3616 is a clean multiple of 32, and
finer granularity is what makes a 5M window reachable at all.

### A note on `ff_expansion` semantics

`ff_expansion` is the literal multiple of `d_model` used as the SwiGLU hidden
width. It is **not** capacity-matched to a stock 4× GELU MLP: SwiGLU spends three
matrices (gate/up/down) where GELU spends two, so at equal expansion a SwiGLU
block is ~1.5× the parameters. For parity with 4× GELU, set `ff_expansion ≈ 2.67`.
3.53 is close to the spec's 3.5, which I kept deliberately.

---

## 2. Positional embeddings (added after review)

**An earlier version of this model had no positional embeddings, on the reasoning
that a pure-SFT empathy model never extrapolates. That was wrong**, and it was the
most serious error in the first pass.

Causal self-attention is permutation-*equivariant* over its input sequence.
Without positional information the model computes an **identical** representation
for every reordering of the same tokens — "dog bites man" and "man bites dog" are
literally indistinguishable. I justified this as an acceptable tradeoff; it is not
a tradeoff at all, it is a defect, and a fatal one for a model whose whole purpose
is reading conversational turn order.

The model now uses learned absolute positions, GPT-2 style (`wpe`,
`block_size × d_model` = 2.1M parameters), added to the token embedding before the
first block.

`src/model.py` asserts this is actually wired up, by checking that reversing a
token sequence changes the output — the exact test that would have caught the
original bug.

### The cost: context is hard-capped at `block_size`

Absolute positions cannot wrap. A prompt of length P can generate at most
`block_size - P` further tokens; with `block_size: 2048`, a 1900-token prompt gets
148 more. This is the standing trade-off of absolute embeddings and precisely why
RoPE and ALiBi exist.

`generate()` checks the budget up front and raises with an actionable message
rather than dying mid-generation on a position error. If you need long-form
generation, **switch to RoPE** (~20 lines, zero parameters) rather than raising
`block_size` blindly.

`forward()` therefore also takes `start_pos`, the absolute position of `idx[:,0]`.
`generate()` tracks it as the window slides. Getting this wrong makes every new
token silently re-read as position 0.

---

## 3. Training defaults not specified in the brief

Chosen for a 497M model on T4 x2, all env-overridable:

| Setting | Value | Reasoning |
|---|---|---|
| LR | 3e-4 → 3e-5 cosine | Standard for a model this size; linear warmup over 200 steps |
| AdamW β₂ | 0.95 | More stable than 0.999 at this LR; GPT-2 precedent |
| Weight decay | 0.1 | Applied to matrices only — see below |
| Batch | 8/GPU × accum 2 × 2 GPUs | ~65k tokens per optimizer step |
| Clip | 1.0 | Empathetic SFT batches can spike loss |
| `torch.compile` | **off** | ~2× step time after a 1-min warmup; on a 12h session with frequent checkpoints the compile cost rarely pays back |

**Weight decay excludes LayerNorm, biases, and the embedding.** Decaying gains
shrinks them toward zero every step, which is a well-known way to hurt
convergence. `build_optimizer` splits into two param groups by tensor rank.

**Gradient clipping happens after `scaler.unscale_()` and before
`scaler.step(optimizer)`** — the only correct ordering. Clipping scaled gradients
is meaningless; the scaler skips the step on inf/nan and you must not let the
clip see those either.

---

## 4. DDP checkpoint barrier

`_checkpoint` does the save/upload work on rank 0 only, but **every rank must
reach the `all_reduce`**. An earlier version early-returned on non-zero ranks
before the collective — rank 0 then blocked forever on a partner that had already
left, and the job died silently at the first checkpoint. On Kaggle that means
losing the entire session with no error.

The guard is `test_checkpoint_barrier_not_guarded`, which parses
`train_ddp.py`'s AST and fails if the `all_reduce` is ever nested under
`if is_main()`. I mutation-tested it by injecting a broken `_checkpoint` and
confirming the check goes red — a guard that cannot fail is worse than none.

---

## 5. Checkpoint format and weight tying

`save_and_push_checkpoint` writes a single `.pt` containing model + optimizer +
step + config.

**Weight tying must survive serialization.** `state_dict()` returns `detach()`
views — distinct objects sharing one storage. A naive `{k: v.cpu()}` allocates a
fresh tensor per key, which *splits the tie* into two copies: +206 MB per
checkpoint here, and a model that silently loses weight tying when resumed.
`_state_dict_to_cpu` dedupes by storage pointer so each tied weight is stored
exactly once. Covered by `test_weight_tying_survives_save`, which round-trips
through a real `torch.save`/`load` and checks the reloaded model reproduces
logits and is still tied.

`causal_mask` is a non-persistent buffer, so it is correctly absent from the
state dict and `strict=True` loading still works.

Uploads are best-effort by design: a failed push logs and returns `None` but never
raises. Without a token the upload is skipped *before* any network call — an
unauthenticated request still burns seconds on a retry ladder and returns a 401
for a repo that may well exist. Rank 0 then all-reduces a failure flag so no rank
proceeds believing a resume point exists.

---

## 6. Hugging Face target

Checkpoints upload to **`siddhu1174/atom-500m`** (`HF_REPO_ID`, overridable).
The Kaggle cell still *clones* from `github.com/BharathDeva1174/atom-500m` —
that is the source repo, a different thing from the upload target. Both are
asserted by tests so a rename cannot silently break one without the other.

---

## 7. What is NOT here

Flagging these explicitly so their absence reads as a decision, not an oversight:

- **No tokenizer/training corpus.** The trainer expects a pre-tokenized
  `data/tokens.npy` (flat `uint16` of GPT-2 ids) and raises a clear error if
  absent. For a model whose entire purpose is empathy, *which* dialogue you
  train on — and its consent and licensing posture — should be decided
  deliberately rather than by scraping whatever is closest.
- **No evaluation harness.** `evaluate()` returns eval loss only; there is no
  empathy-specific metric. Automated empathy scoring is a research problem
  (LLM-judge rubrics are noisy and poorly validated), and a bad proxy metric
  would be worse than none — it would let you optimize the wrong thing while
  believing the numbers.
- **No inference/chat script.** `model.generate()` exists and is tested; no chat
  template or turn-taking wrapper is defined.
- **`prompt_mask` is only wired for the HF streaming path**, and masks a simple
  prefix. It is rejected outright on pre-tokenized arrays, which have no boundary
  information — guessing a prefix would silently train on the wrong tokens.

---

## 8. Verified

`python scripts/integration_test.py` — 50 checks on CPU:

- Tiny training loop converges (loss 6.09 → 0.61 over 30 epochs)
- Gradients flow to the tied embedding and to `wpe`
- Model is permutation-sensitive; `start_pos` shifts the encoding
- Shipped config lands in the 495M–500M band
- `generate()` respects the position budget and preserves the prompt
- Dataset windowing: `y[:-1] == x[1:]`, correct offsets, dtype widening
- LR schedule: warmup peak, monotonic cosine decay, floors at `MIN_LR`
- Optimizer param groups: LayerNorm/embedding excluded from decay
- Deadlock guard and weight-tying round-trip (both mutation-tested)

`python scripts/ddp_smoke_test.py` — a real 2-rank gloo run through the actual
training loop: both ranks exit 0, eval loss 6.64 → 5.97.

**Not verified:** anything requiring real NVIDIA GPUs — FlashAttention dispatch,
NCCL all-reduce, `GradScaler` overflow behavior under actual fp16, and the
Hugging Face upload path. These need the Kaggle run to confirm.
