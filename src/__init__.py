"""Atom-500M — a decoder-only GPT for high-EQ conversational empathy."""

from .model import Atom500M, Block, CausalSelfAttention, SwiGLU, load_config

__all__ = ["Atom500M", "Block", "CausalSelfAttention", "SwiGLU", "load_config"]
__version__ = "0.1.0"
