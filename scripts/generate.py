"""Local inference for Atom-500M checkpoints.

The trainer exists, but until a checkpoint is actually talked to, "eval loss
3.86" means nothing about whether the model listens. This is the smallest
harness that answers that: load a checkpoint, type a situation and what the
person says, read what the model says back.

The model is trained on documents laid out as

    Situation: <why the person is talking>
    Person: <the distressed person>
    Listener: <the model's reply>
    Person: ...

so it is prompted in exactly that shape and expected to produce one
`Listener:` turn. It was never trained to write the other roles, so
generation stops before "Person:" reappears -- continuing the dialogue as
both sides is not something this model knows how to do.

Examples
--------
    python scripts/generate.py --prompt "Situation: lost his job.\\nPerson: I don't know what to do next.\\nListener:"
    python scripts/generate.py --interactive --checkpoint checkpoints/step_1500.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Atom500M, load_config  # noqa: E402

# GPT-2's <|endoftext|>, which terminate_every conversation in the corpus.
EOT = 50256
DEFAULT_CHECKPOINT = "checkpoints/step_3000.pt"

# Roles the model was never trained to speak. Emitting one means it has run
# off the end of its turn, so generation stops there.
DEFAULT_STOPS = ("\nPerson:", "\nSituation:", "\nListener:", "\n\n", "\n")


def build_tokenizer(name: str = "gpt2"):
    """The corpus was tokenized with tiktoken's gpt2 BPE; anything else and
    every id is wrong."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tiktoken is required. pip install tiktoken") from exc
    return tiktoken.get_encoding(name)


def load_model(checkpoint: str, device: torch.device, dtype: torch.dtype):
    """Build the architecture from the shipped config and load its weights.

    strict=True on purpose: a key that does not belong usually means the
    checkpoint came from a different architecture (a block_size change, say),
    and silently dropping the mismatched pieces produces a model that
    generates fluent nonsense instead of an error.
    """
    config = load_config()
    model = Atom500M(config)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"could not load {checkpoint}:\n  {str(exc).splitlines()[0][:200]}\n"
            f"This checkpoint was almost certainly trained with a different "
            f"architecture. Check that configs/atom_500m.json still matches the "
            f"run that produced it (block_size, d_model, n_layer, n_head)."
        ) from exc

    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    return model.to(device=device, dtype=dtype).eval(), step, config


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k then nucleus filtering, in that order."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")
    if 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        # Sort descending, keep the smallest set whose mass reaches top_p.
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        drop = cum - sorted_probs > top_p          # keep the token that crosses
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
        logits = torch.log(probs.clamp_min(1e-12))
    return logits


@torch.no_grad()
def generate_reply(
    model: Atom500M,
    enc,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stops: tuple,
    seed: Optional[int] = None,
) -> str:
    """Sample one Listener turn.

    The model uses learned ABSOLUTE positions, so the sequence is hard-capped
    at block_size: the sliding window slides, but the position of the first
    token in the window must be passed every step or the model silently
    re-reads older text as position 0.
    """
    if seed is not None:
        torch.manual_seed(seed)

    ids = enc.encode_ordinary(prompt)
    budget = model.block_size - len(ids)
    if budget <= 0:
        raise SystemExit(
            f"prompt is {len(ids)} tokens; block_size is {model.block_size}. "
            f"Shorten the conversation history (absolute positions cannot wrap)."
        )
    max_new_tokens = min(max_new_tokens, budget)

    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    out: List[int] = []
    text = ""

    for _ in range(max_new_tokens):
        window = idx[:, -model.block_size:]
        start_pos = idx.size(1) - window.size(1)
        logits = model(window, start_pos=start_pos)[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            # Penalize anything already said, prompt included -- an empathy
            # model left unchecked will happily repeat "I'm sorry to hear
            # that" until the turn budget runs out.
            for tid in set(ids + out):
                v = logits[0, tid]
                logits[0, tid] = (v / repetition_penalty if v > 0
                                   else v * repetition_penalty)

        if temperature <= 0.0:
            nxt = int(torch.argmax(logits, dim=-1))
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1))

        if nxt == EOT:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=idx.device)], dim=1)

        text = enc.decode(out)
        for s in stops:
            if s in text:
                text = text.split(s)[0]
                return text.strip()

    return text.strip()


def format_turn(situation: str, person: str) -> str:
    """The corpus's own layout, so the prompt is in-distribution."""
    parts = []
    if situation:
        parts.append(f"Situation: {situation.strip()}")
    parts.append(f"Person: {person.strip()}")
    parts.append("Listener:")
    return "\n".join(parts)


def run_once(model, enc, args, situation: str = "", history: List[str] = None):
    """Prompt the model with a situation + what the person said."""
    person = args.prompt.strip() if args.prompt else input("Person: ").strip()
    turn = f"Person: {person}"
    context = format_turn(situation, turn) if not history else "\n".join(
        history + [turn, "Listener:"]
    )
    t0 = time.time()
    reply = generate_reply(
        model, enc, context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stops=tuple(args.stop),
        seed=args.seed,
    )
    dt = time.time() - t0
    print(f"\nListener: {reply}")
    print(f"  [{len(reply.split())} words, {dt:.1f}s]")
    return turn, reply


def interactive(model, enc, args):
    """Multi-turn testing with the weights loaded once.

    The conversation accumulates in corpus format so later turns have the same
    shape the model saw in training. `/reset` clears it, `/situation <text>`
    sets the header, `/quit` leaves.
    """
    situation = ""
    history: List[str] = []
    print("\nInteractive mode. Commands: /situation <text>, /reset, /quit")
    print("Type what the Person says; the model replies as Listener.\n")

    while True:
        try:
            line = input("Person: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/quit"):
            break
        if line.startswith("/reset"):
            history, situation = [], ""
            print("[reset]")
            continue
        if line.startswith("/situation"):
            situation = line[len("/situation"):].strip()
            print(f"[situation] {situation}")
            continue

        turn = f"Person: {line}"
        context = format_turn(situation, turn) if not history else "\n".join(
            history + [turn, "Listener:"]
        )
        reply = generate_reply(
            model, enc, context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stops=tuple(args.stop),
            seed=args.seed,
        )
        print(f"Listener: {reply}\n")
        history += ([f"Situation: {situation}"] if situation and not history else [])
        history += [turn, f"Listener: {reply}"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a local Atom-500M checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"path to a .pt checkpoint (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--prompt", default="",
                    help="what the Person says; omit for a bare situation prompt")
    ap.add_argument("--situation", default="",
                    help="Situation: line, e.g. 'just lost his job'")
    ap.add_argument("--interactive", action="store_true",
                    help="multi-turn loop; weights load once")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = greedy; lower is warmer and more consistent")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--stop", action="append", default=None,
                    help=f"stop string (repeatable); default {DEFAULT_STOPS}")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"no checkpoint at {args.checkpoint}\n"
            f"Download one first:\n"
            f"  python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('siddhu1174/atom-500m','checkpoints/step_3000.pt',"
            f"local_dir='.')\""
        )
    if args.stop is None:
        args.stop = list(DEFAULT_STOPS)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto" else "cpu"
    )
    dtype = {
        "auto": torch.float16 if device.type == "cuda" else torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"loading {args.checkpoint} on {device.type} ({dtype}) ...")
    model, step, config = load_model(args.checkpoint, device, dtype)
    enc = build_tokenizer()
    print(f"{model.num_parameters():,} parameters"
          + (f" | trained to step {step}" if step is not None else "")
          + f" | sampling T={args.temperature} top_p={args.top_p} top_k={args.top_k}"
          + f" rep={args.repetition_penalty}")

    if args.interactive:
        interactive(model, enc, args)
        return

    if not args.prompt and not sys.stdin.isatty():
        raise SystemExit("no --prompt given and stdin is not a terminal; "
                         "pass --prompt or --interactive")

    person = args.prompt or input("Person: ").strip()
    context = format_turn(args.situation, person)
    print(f"\n--- prompt ---\n{context}\n--- reply ---")
    reply = generate_reply(
        model, enc, context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stops=tuple(args.stop),
        seed=args.seed,
    )
    print(f"Listener: {reply}")


if __name__ == "__main__":
    main()