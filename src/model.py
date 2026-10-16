"""Atom-500M — a 500M-parameter decoder-only GPT built for high-EQ conversation.

Architecture notes
------------------
Pre-LayerNorm decoder-only transformer (GPT-2 lineage):

    x = x + Attn(LN(x))
    x = x + FFN(LN(x))

Three deliberate departures from a stock nanoGPT config:

1. SwiGLU feed-forward (PaLM / LLaMA style) instead of the 4x GELU MLP. A gated
   MLP is empirically better per-parameter, so the hidden width can be trimmed
   for the savings.

2. Learned absolute positional embeddings (`wpe`), GPT-2 style. These are not
   optional: causal self-attention alone is permutation-*equivariant* over its
   input sequence, so without position information the model cannot distinguish
   "dog bites man" from "man bites dog" — it computes the same representation for
   every reordering of the same tokens. For a model whose entire purpose is
   reading conversational turn order, that is fatal, not a tradeoff.

3. Tied token-embedding / output-head weights, so the 51M-parameter embedding
   matrix is shared rather than duplicated.

Because positions are absolute, `forward` takes a `start_pos` offset and
`generate` tracks it — see those docstrings. Getting that wrong makes
incremental decoding silently read every new token as position 0.

Parameter budget: the config targets 495M-500M. `python src/model.py` prints the
measured count and a full component breakdown on every run, so the number is
never a guess.
"""

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "atom_500m.json",
)

# `ff_expansion` is the literal multiple of d_model used as the SwiGLU hidden
# width, rounded to a multiple of 32 for Tensor Core / cuBLAS friendliness.
# 32 is a preference, not a hardware requirement: every multiple of 8 works, and
# finer granularity is what makes it possible to land the total parameter count
# inside a target band (see the budget arithmetic in NOTES.md).
#
# Note that SwiGLU spends 3 matrices (gate/up/down) where a GELU MLP spends 2,
# so at equal expansion a SwiGLU block is ~1.5x the parameters -- intended.
_HIDDEN_MULTIPLE = 32


def _round_to_multiple(n: int, multiple: int) -> int:
    """Round `n` to the nearest multiple of `multiple` (min one full block)."""
    return max(multiple, int(round(n / multiple)) * multiple)


def load_config(path: Optional[str] = None) -> dict:
    """Load the hyperparameter dict that `Atom500M` is constructed from."""
    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses `F.scaled_dot_product_attention(is_causal=True)` so PyTorch dispatches to
    FlashAttention-2 on CUDA (Ampere+), the memory-efficient kernel on older
    cards, and a math fallback on CPU. No explicit mask tensor is materialized.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_head = config["n_head"]
        self.d_model = config["d_model"]
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        self.head_dim = self.d_model // self.n_head

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        # Bias on the output projection keeps the residual stream clean; nanoGPT
        # omits it but it is a free win at this width.
        self.out_proj.bias = nn.Parameter(torch.zeros(self.d_model))

        # Buffers, not parameters: survive .to(device) and get written into the
        # state dict without showing up in the parameter count.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config["block_size"], config["block_size"], dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        assert T <= self.causal_mask.size(0), (
            f"sequence length {T} exceeds block_size {self.causal_mask.size(0)}"
        )

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # (B, n_head, T, head_dim); is_causal=True gives FlashAttention on CUDA.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)


class SwiGLU(nn.Module):
    """Gated feed-forward network:  down( silu(gate(x)) * up(x) )."""

    def __init__(self, config: dict):
        super().__init__()
        self.hidden = _round_to_multiple(
            int(config["ff_expansion"] * config["d_model"]), _HIDDEN_MULTIPLE
        )
        self.gate_proj = nn.Linear(config["d_model"], self.hidden, bias=False)
        self.up_proj = nn.Linear(config["d_model"], self.hidden, bias=False)
        self.down_proj = nn.Linear(self.hidden, config["d_model"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-LN transformer block: attention and SwiGLU, each with a residual."""

    def __init__(self, config: dict):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config["d_model"])
        self.attn = CausalSelfAttention(config)
        self.ln_ffn = nn.LayerNorm(config["d_model"])
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class Atom500M(nn.Module):
    """Decoder-only transformer for high-EQ conversational empathy.

    Trained strictly on empathetic dialogue -- no code, math, or technical
    domain logic.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = dict(config) if config is not None else load_config()

        self.config = config
        self.block_size = config["block_size"]
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]

        self.wte = nn.Embedding(config["vocab_size"], config["d_model"])
        # Learned absolute positions, GPT-2 style. Causal attention restricts
        # *which* tokens a position may attend to, but without this the model
        # still cannot tell where in the sequence any token sits.
        self.wpe = nn.Embedding(config["block_size"], config["d_model"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

        # Gradient checkpointing. When on, each block's inner activations are
        # discarded after the forward pass and recomputed during backward,
        # trading ~30% extra compute for a large cut in activation VRAM.
        # Defaults OFF: this is a memory/speed trade the caller should choose,
        # and enabling it silently would make the default path ~30% slower.
        self.grad_checkpointing = bool(config.get("grad_checkpointing", False))

        # Weight tying: one 51.5M-parameter embedding matrix serves as both input
        # embedding and output classifier. Applied after init so that the tied
        # matrix is initialized from the (smaller-std) embedding init.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual-output projections (attn.out_proj,
        # ffn.down_proj) — stabilizes deep pre-LN stacks.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"]))

    # -- init ---------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- helpers ------------------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Parameter count. Tied weights appear once because `lm_head.weight`
        and `wte.weight` are the same `Parameter` object."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample a continuation. Greedy-ish by default: empathy replies should be
        warm and consistent, not creative.

        **Context is hard-capped at `block_size` total tokens.** Learned absolute
        positions cannot wrap, so a prompt of length P can generate at most
        `block_size - P` more tokens. This is the standing trade-off of absolute
        embeddings and the reason RoPE/ALiBi exist; if you need long generation,
        switch the position scheme rather than raising `block_size` blindly.

        The check is done up front so an over-long request fails immediately with
        a clear message instead of mid-generation on a cryptic position error.
        """
        if idx.size(1) + max_new_tokens > self.block_size:
            room = self.block_size - idx.size(1)
            raise ValueError(
                f"cannot generate {max_new_tokens} tokens from a "
                f"{idx.size(1)}-token prompt: learned absolute positions cap the "
                f"sequence at block_size={self.block_size}, leaving {room} "
                f"token(s) of room. Reduce max_new_tokens, or replace the "
                f"absolute position scheme with RoPE for unbounded generation."
            )

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            # Where idx_cond begins within the original sequence. Absolute
            # positions mean the window's first token is not position 0 once the
            # context has filled; passing start_pos keeps the encoding honest.
            start_pos = idx.size(1) - idx_cond.size(1)
            logits = self(idx_cond, start_pos=start_pos)[:, -1, :]
            if temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
        return idx

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """idx: (B, T) int64 token ids. Returns loss if targets given, else logits.

        `targets` are ALREADY aligned with `idx`: targets[:, i] is the token
        that follows idx[:, i], the same length T. The datasets build exactly
        that (x = chunk[:-1], y = chunk[1:]), so the loss is computed straight
        through -- logits[:, i] against targets[:, i].

        An earlier version shifted again here, so at position i it trained the
        model to predict targets[:, i+1] -- the token TWO ahead. Every
        checkpoint from that run produces word salad when sampled normally:
        it scored ~3.9 on its own objective and 8.7 (near-random) on real
        next-token prediction. Do not shift targets again here; if a caller
        hands over unshifted targets, that caller is the bug.

        `start_pos` is the absolute position of `idx[:, 0]`. It is 0 for training
        and for a fresh prompt, and grows only when `generate` slides its window
        past `block_size`.
        """
        B, T = idx.size()
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )
        assert idx.dtype == torch.long, f"expected int64 token ids, got {idx.dtype}"

        # Positional ids are materialized per call rather than cached, so the
        # device always matches idx (a cached tensor would strand the model on
        # the CPU device if idx moved to GPU). arange allocates T int64 values.
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        if start_pos + T > self.block_size:
            raise ValueError(
                f"positions {start_pos}..{start_pos + T} exceed block_size "
                f"{self.block_size}; absolute positions cannot wrap. Shorten the "
                f"context or raise block_size."
            )

        x = self.wte(idx) + self.wpe(pos)  # (B, T, C)

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                # use_reentrant=False is required, not stylistic. The reentrant
                # implementation does not support a nested/autograd structure
                # where inputs are passed positionally, and silently produces
                # no gradients for some parameter patterns. The non-reentrant
                # version also lets us checkpoint with no grad-requiring inputs
                # and handles autocast correctly under mixed precision.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, V) — tied to wte.weight

        if targets is None:
            return logits

        # targets[:, i] is already the token after idx[:, i] (see docstring).
        # Shifting here as well trained a two-token-ahead predictor.
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,  # lets the dataset mask out prompt tokens
        )
        return loss


if __name__ == "__main__":
    cfg = load_config()
    model = Atom500M(cfg)

    n_params = model.num_parameters()
    n_tied = model.num_parameters(non_embedding=True)
    print("=" * 62)
    print("Atom-500M")
    print("=" * 62)
    print(f"  config             : {cfg}")
    print(f"  SwiGLU hidden width: {model.blocks[0].ffn.hidden}  "
          f"({cfg['ff_expansion']} * d_model)")
    print(f"  positional embeds  : learned absolute, {cfg['block_size']} x {cfg['d_model']}")
    print(f"  tied lm_head       : {model.lm_head.weight is model.wte.weight}")
    print("-" * 62)
    print(f"  total parameters   : {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  non-embedding      : {n_tied:,} ({n_tied / 1e6:.2f}M)")
    print(f"  tied embedding     : {model.wte.weight.numel():,}")
    print("-" * 62)

    # Dummy forward pass: logits shape, plus a backward pass to prove the graph.
    idx = torch.randint(0, cfg["vocab_size"], (2, 128))
    targets = torch.randint(0, cfg["vocab_size"], (2, 128))
    logits = model(idx)
    loss = model(idx, targets)
    loss.backward()

    print(f"  dummy input shape  : {tuple(idx.shape)}")
    print(f"  logits shape       : {tuple(logits.shape)}")
    print(f"  cross-entropy loss : {loss.item():.4f}  (ln(V) = {math.log(cfg['vocab_size']):.4f})")

    # The loss must align logits[:, i] with targets[:, i]. The datasets already
    # align y to x (y = chunk[1:]); shifting again inside forward trains a
    # two-token-ahead predictor that scores well on its own objective and
    # generates word salad when sampled normally. Pin the arithmetic.
    with torch.no_grad():
        aligned = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg["vocab_size"]), targets.reshape(-1))
        shifted = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg["vocab_size"]),
            targets[:, 1:].reshape(-1))
    ok = abs(loss.item() - aligned.item()) < 1e-5
    print(f"  loss alignment     : {'logits[:,i] vs targets[:,i]' if ok else 'MISALIGNED'}"
          f"  (a second shift would give {shifted.item():.4f})")
    assert ok, "forward shifts targets that the dataset already aligned"
    print("=" * 62)

    # --- Parameter budget: how the shipped config splits down --------------
    emb = model.wte.weight.numel()
    pos = model.wpe.weight.numel()
    attn_per_layer = model.blocks[0].attn.q_proj.weight.numel() * 3 + model.blocks[0].attn.out_proj.weight.numel()
    ffn_per_layer = sum(p.numel() for p in model.blocks[0].ffn.parameters())
    rest = n_params - emb - pos - (attn_per_layer + ffn_per_layer) * cfg["n_layer"]
    print("  budget breakdown")
    print(f"    token embedding (tied)  {emb:>12,}")
    print(f"    position embedding      {pos:>12,}")
    print(f"    attention  x {cfg['n_layer']:<2} layers   {attn_per_layer * cfg['n_layer']:>12,}")
    print(f"    SwiGLU     x {cfg['n_layer']:<2} layers   {ffn_per_layer * cfg['n_layer']:>12,}")
    print(f"    LayerNorms + biases     {rest:>12,}")
    print("=" * 62)

    assert 495e6 <= n_params <= 500e6, (
        f"expected 495M-500M parameters, got {n_params:,} "
        f"({n_params / 1e6:.2f}M). See NOTES.md for the sizing arithmetic."
    )
    print(f"  within 495M-500M target: YES ({n_params / 1e6:.2f}M)")
    print("=" * 62)

    # Causality check: perturbing token t must not alter logits at positions < t.
    probe = Atom500M(cfg).eval()
    x = torch.randint(0, cfg["vocab_size"], (1, 64))
    x2 = x.clone()
    x2[0, 40] = (x2[0, 40] + 1) % cfg["vocab_size"]
    with torch.no_grad():
        d = (probe(x)[:, :40] - probe(x2)[:, :40]).abs().max().item()
    print(f"  causality (max logit drift at t<40): {d:.2e}  {'OK' if d == 0.0 else 'LEAK'}")
    assert d == 0.0, "causality violated: future tokens leaked into earlier positions"

    # Position sensitivity: without wpe the model is permutation-equivariant, so
    # swapping two tokens must change the output. This is the check that would
    # have caught the missing positional embedding.
    with torch.no_grad():
        a = torch.tensor([[10, 20, 30, 40]])
        b = torch.tensor([[40, 30, 20, 10]])  # same tokens, reversed order
        d_perm = (probe(a) - probe(b)).abs().max().item()
    print(f"  permutation sensitivity        : {d_perm:.4f}  "
          f"{'OK' if d_perm > 1e-6 else 'DEGENERATE (no position signal)'}")
    assert d_perm > 1e-6, "model is permutation-invariant: wpe is not wired up"

    # start_pos must actually shift the encoding, or generate() slides the
    # window while silently re-reading every token as position 0.
    with torch.no_grad():
        d_pos = (probe(x, start_pos=0) - probe(x, start_pos=7)).abs().max().item()
    print(f"  start_pos sensitivity          : {d_pos:.4f}  "
          f"{'OK' if d_pos > 1e-6 else 'IGNORED'}")
    assert d_pos > 1e-6, "start_pos has no effect; generate() would mis-encode"

    # Positions past block_size must raise rather than silently wrap.
    try:
        probe(x[:, :32], start_pos=cfg["block_size"] - 8)
        print("  start_pos overflow             : NOT GUARDED")
    except ValueError:
        print("  start_pos overflow             : raises ValueError  OK")

    print(f"OK — {n_params / 1e6:.2f}M parameters, forward + backward clean.")
