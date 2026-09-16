"""Memory-bounded acoustic flow matching with one AR prefill per original chunk.

The reference midpoint solver, full-song CPU FP32 noise draw,
boundary positions, and original context chunks are preserved.
Attention query tiling changes temporary storage, never the visible key set.
"""
from __future__ import annotations

import math
from .model import attention as model_attention
from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import torch
import torch.nn.functional as F

from .protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def song_chunks(prefix, codec, seed, context=CONTEXT):
    """Draw the complete noise tensor once, then take views at historical cuts."""
    prefix = _integers(prefix, "prefix")
    codec = _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= CONTEXT:
        raise ValueError(f"context must be an integer in 1..{CONTEXT}")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [Chunk(prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END], noise[a:b].clone())
            for a, b in ranges]


def attention(q, k, v, causal=False):
    return model_attention(q[None], k[None], v[None], causal)[0]


class CachedNAR:
    """One original acoustic chunk; AR prefix KV is invariant during the ODE."""

    def __init__(self, model, chunk: Chunk, conditioning=None, zero_conditioning=None, condition_scale=1.0):
        self.model, self.chunk = model, chunk
        self.conditioning, self.zero_conditioning, self.condition_scale = conditioning, zero_conditioning, condition_scale
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(chunk.noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config.max_position_embeddings:
            raise ValueError("Original acoustic chunk exceeds the model context")
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _attention(self, q, k, v, causal=False):
        return attention(q, k, v, causal=causal)

    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            self.cache.append((k[0], v[0]))
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    def velocity(self, state, raw_t):
        value = self._velocity(state, raw_t, self.conditioning)
        if self.zero_conditioning is not None and self.condition_scale != 1.0:
            zero = self._velocity(state, raw_t, self.zero_conditioning)
            value = zero + self.condition_scale * (value - zero)
        return value

    def _velocity(self, state, raw_t, conditioning):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for index, (layer, (ar_k, ar_v)) in enumerate(zip(model.model.layers, self.cache)):
            if conditioning is not None and index in conditioning:
                x = x + conditioning[index].to(x.dtype)
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None):
        """Solve a chunk, reporting each submitted midpoint step without syncing.

        CUDA work may still be executing when ``on_progress`` runs. The existing
        CPU result transfer completes that work before this method returns.
        Callback exceptions propagate to the caller.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self.chunk.noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = 20.0 if t == 1 else max(-20.0, min(20.0, math.log(t / (1 - t))))
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            raw_mid = max(-20.0, min(20.0, math.log((t - dt / 2) / (1 - t + dt / 2))))
            state = state - self.velocity(mid, raw_mid) * dt
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None
        self.conditioning = self.zero_conditioning = None


def synthesize(model, prefix, codec, seed, steps=32, cancelled=None, on_progress=None, *, conditioner=None, condition=None, condition_scale=1.0):
    if conditioner is not None and (condition is None or tuple(condition.shape) != (len(codec), 64)):
        raise ValueError("Source conditioning must have one 64-channel latent per semantic frame")
    chunks = song_chunks(prefix, codec, seed)
    output = []
    offset = 0
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        conditioned = zero = None
        if conditioner is not None:
            weight = next(conditioner.parameters())
            source = condition[offset:offset + len(chunk.noise)].to(device=weight.device, dtype=weight.dtype)
            conditioned = conditioner(source)
            if condition_scale != 1.0:
                zero = conditioner(torch.zeros_like(source))
        engine = CachedNAR(model, chunk, conditioned, zero, condition_scale)
        try:
            def progress(completed, total):
                if on_progress is not None:
                    on_progress(chunk_index * total + completed, total * len(chunks))
            output.append(engine.solve(steps, cancelled, progress))
        finally:
            engine.close()
        offset += len(chunk.noise)
    return torch.cat(output, dim=0)
