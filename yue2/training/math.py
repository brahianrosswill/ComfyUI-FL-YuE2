"""Differentiable YuE2 training paths; inference interfaces remain unchanged."""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model import attention
from ..protocol import MUSIC_END


class TokenHead(nn.Module):
    def __init__(self, width=512, layers=8, heads=8, window=512, input_dim=1024, vocab=32768):
        super().__init__()
        self.inp = nn.Linear(input_dim, width)
        self.pos = nn.Parameter(torch.empty(1, window, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, x):
        return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + self.scale * ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


@contextmanager
def adapter_scale(model, scale):
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    previous = [module.scale for module in modules]
    for module in modules:
        module.scale = scale
    try:
        yield
    finally:
        for module, value in zip(modules, previous):
            module.scale = value


def qkv(module, x, cos, sin):
    batch, length, _ = x.shape
    q = module.q_proj(x).reshape(batch, length, module.num_heads, module.head_dim)
    k = module.k_proj(x).reshape(batch, length, module.num_kv_heads, module.head_dim)
    v = module.v_proj(x).reshape(batch, length, module.num_kv_heads, module.head_dim)
    # comfy-kitchen 0.2.31's fused split-half operation has no autograd formula.
    q, k = module.q_norm(q), module.k_norm(k)
    cos, sin = cos.unsqueeze(2).to(q.dtype), sin.unsqueeze(2).to(q.dtype)
    def rotate(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return rotate(q), rotate(k), v


def ar_layer(layer, x, cos, sin):
    q, k, v = qkv(layer.self_attn, layer.input_layernorm(x), cos, sin)
    x = x + layer.self_attn.o_proj(attention(q, k, v, True).flatten(2))
    return x + layer.mlp(layer.post_attention_layernorm(x)), k, v


def hidden(model, ids, checkpoint_layers=True):
    x = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(torch.arange(x.shape[1], device=x.device)[None])
    for layer in model.model.layers:
        x, _, _ = checkpoint(ar_layer, layer, x, cos, sin, use_reentrant=False) if checkpoint_layers else ar_layer(layer, x, cos, sin)
    return model.model.norm(x)


def ar_loss(model, ids, prefix_length, cursor=None, cursor_head=None, checkpoint_layers=True):
    h = hidden(model, ids, checkpoint_layers)[0]
    audio_h, target = h[prefix_length - 1:-1], ids[0, prefix_length:]
    loss = torch.zeros((), device=h.device)
    for start in range(0, len(target), 256):
        def ce(value, labels):
            return F.cross_entropy(model.lm_head(value).float(), labels, reduction="sum")
        loss = loss + checkpoint(ce, audio_h[start:start + 256], target[start:start + 256], use_reentrant=False)
    loss = loss / len(target)
    cursor_loss = None
    if cursor is not None:
        begin, end, targets = cursor
        count = min(len(targets), len(audio_h))
        query = cursor_head(audio_h[:count].float())
        scores = query @ h[begin:end].float().T / math.sqrt(h.shape[-1])
        cursor_loss = -(scores.log_softmax(-1) * targets[:count]).sum(-1).mean()
    return loss, cursor_loss


def ar_joint_loss(model, ids, prefix_length, kl_weight, checkpoint_layers=True):
    with torch.set_grad_enabled(False), adapter_scale(model, 0.0):
        base = hidden(model, ids, False)[0, prefix_length - 1:-1].detach()
    adapted = hidden(model, ids, checkpoint_layers)[0, prefix_length - 1:-1]
    target = ids[0, prefix_length:]
    ce_loss = torch.zeros((), device=adapted.device)
    kl_loss = torch.zeros((), device=adapted.device)
    for start in range(0, len(target), 256):
        stop = start + 256
        def losses(value, base_value, labels):
            logits = model.lm_head(value).float()
            ce = F.cross_entropy(logits, labels, reduction="sum")
            if not kl_weight:
                return ce, ce.new_zeros(())
            with torch.set_grad_enabled(False):
                base_logp = model.lm_head(base_value).float().log_softmax(-1)
            kl = F.kl_div(logits.log_softmax(-1), base_logp, reduction="sum", log_target=True)
            return ce, kl
        ce, kl = checkpoint(losses, adapted[start:stop], base[start:stop], target[start:stop], use_reentrant=False)
        ce_loss = ce_loss + ce
        kl_loss = kl_loss + kl
    return ce_loss / len(target), kl_loss / len(target)


def nar_layer(layer, x, ar_k, ar_v, cos, sin):
    q, k, v = qkv(layer.nar_self_attn, layer.nar_input_layernorm(x), cos, sin)
    x = x + layer.nar_self_attn.o_proj(attention(q, torch.cat((ar_k, k), 1), torch.cat((ar_v, v), 1)).flatten(2))
    return x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))


def flow_loss(model, prefix, codec_embeddings, target, t, noise, train_head, training=True, conditioning=None):
    backbone = model.model
    device = target.device
    pre = backbone.embed_tokens(torch.tensor(prefix, device=device))
    end = backbone.embed_tokens(torch.tensor([MUSIC_END], device=device))
    x = torch.cat((pre, codec_embeddings.to(pre.dtype), end), 0)[None]
    length, frames = x.shape[1], len(target)
    cos, sin = backbone.rotary_emb(torch.arange(length, device=device)[None])
    cache = []
    with torch.set_grad_enabled(training and train_head):
        for layer in backbone.layers:
            x, k, v = checkpoint(ar_layer, layer, x, cos, sin, use_reentrant=False) if training and train_head else ar_layer(layer, x, cos, sin)
            cache.append((k, v))
    cos, sin = backbone.rotary_emb(torch.arange(length, length + frames + 2, device=device)[None])
    positions = torch.arange(frames + 2, device=device).clamp(max=model.config.max_latent_frames - 1)
    shifted = model._shift_t_value(math.log(t / (1 - t)), device, pre.dtype)
    x = model.vae2llm(F.pad(t * noise + (1 - t) * target, (0, 0, 1, 1))[None]).to(pre.dtype)
    x = x + model.time_embedder(shifted.expand(frames + 2))[None] + model.latent_pos_embed(positions)[None]
    for index, (layer, (k, v)) in enumerate(zip(backbone.layers, cache)):
        if conditioning is not None and index in conditioning:
            x = x + conditioning[index].to(x.dtype)
        x = checkpoint(nar_layer, layer, x, k, v, cos, sin, use_reentrant=False) if training else nar_layer(layer, x, k, v, cos, sin)
    return F.mse_loss(model.llm2vae(backbone.norm(x))[0, 1:-1].float(), noise - target)
