"""Differentiable YuE2 training paths; inference interfaces remain unchanged."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model import attention


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

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


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
