import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
import comfy.ops
import comfy_kitchen
from comfy.ldm.modules.attention import optimized_attention_for_device


def attention(q, k, v, causal=False):
    # [B,T,H,D]; query tiles bound causal-mask storage without hiding past keys.
    batch, length, heads, dim = q.shape
    selected = optimized_attention_for_device(q.device, mask=causal)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    outputs = []
    block = 256 if causal else length
    for start in range(0, length, block):
        end = min(start + block, length)
        key_end = end if causal else k.shape[2]
        mask = None
        if causal:
            visible = torch.arange(key_end, device=q.device)[None] <= torch.arange(start, end, device=q.device)[:, None]
            mask = torch.zeros((end - start, key_end), dtype=q.dtype, device=q.device).masked_fill_(~visible, -torch.inf)
        outputs.append(selected(q[:, :, start:end], k[:, :, :key_end], v[:, :, :key_end],
                                heads, mask=mask, skip_reshape=True, enable_gqa=heads != k.shape[1]))
    return torch.cat(outputs, dim=1).reshape(batch, length, heads, dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.head_dim, self.base = head_dim, base

    def forward(self, positions):
        inverse = 1.0 / self.base ** (torch.arange(0, self.head_dim, 2, device=positions.device, dtype=torch.float32) / self.head_dim)
        angles = positions.float().unsqueeze(-1) * inverse
        return angles.cos(), angles.sin()


class Attention(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = operations.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = operations.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = operations.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = operations.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def project_qkv(self, x, cos, sin):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        rotation = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(*cos.shape, 2, 2).unsqueeze(2).to(q.dtype)
        with comfy.ops.CastBiasWeightContext(self.q_norm, q, offloadable=True) as (q_weight, _):
            with comfy.ops.CastBiasWeightContext(self.k_norm, k, offloadable=True) as (k_weight, _):
                q, k = comfy_kitchen.rms_rope_split_half(q, k, rotation, q_weight, k_weight, self.q_norm.eps)
        return q, k, v


class MLP(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.gate_proj = operations.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = operations.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = operations.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, operations)
        self.nar_input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nar_self_attn = Attention(config, operations)
        self.post_attention_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config, operations)
        self.nar_pre_mlp_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nar_mlp = MLP(config, operations)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, operations):
        super().__init__()
        self.mlp = nn.Sequential(operations.Linear(256, hidden_size), nn.SiLU(), operations.Linear(hidden_size, hidden_size))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
        angles = t.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding.to(t.dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, frames, hidden_size):
        super().__init__()
        self.register_buffer("pe", torch.empty(frames, hidden_size))

    def forward(self, positions):
        return self.pe[positions]


class Backbone(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, operations) for _ in range(config.num_hidden_layers))
        self.norm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)


class StaticKVCache:
    def __init__(self, num_layers, batch_size, num_kv_heads, max_seq_len, head_dim, dtype, device):
        self.position = 0
        self.keys = [torch.empty(batch_size, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
        self.values = [torch.empty_like(value) for value in self.keys]

    def update(self, k, v, layer):
        end = self.position + k.shape[1]
        self.keys[layer][:, self.position:end].copy_(k)
        self.values[layer][:, self.position:end].copy_(v)
        if layer == len(self.keys) - 1:
            self.position = end
        return self.keys[layer][:, :end], self.values[layer][:, :end]


class YuE2Model(nn.Module):
    def __init__(self, config, operations=comfy.ops.manual_cast):
        super().__init__()
        self.config = SimpleNamespace(**config)
        c = self.config
        self.model = Backbone(c, operations)
        self.lm_head = operations.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.llm2vae = operations.Linear(c.hidden_size, c.latent_dim)
        self.vae2llm = operations.Linear(c.latent_dim, c.hidden_size)
        self.time_embedder = TimestepEmbedder(c.hidden_size, operations)
        self.latent_pos_embed = AudioPositionEmbedding(c.max_latent_frames, c.hidden_size)

    def forward(self, input_ids, past_key_values, logits_to_keep=1):
        x = self.model.embed_tokens(input_ids)
        positions = torch.arange(past_key_values.position, past_key_values.position + x.shape[1], device=x.device)[None]
        cos, sin = self.model.rotary_emb(positions)
        causal = past_key_values.position == 0 and x.shape[1] > 1
        for index, layer in enumerate(self.model.layers):
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            k, v = past_key_values.update(k, v, index)
            x = x + layer.self_attn.o_proj(attention(q, k, v, causal).flatten(2))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = self.lm_head(self.model.norm(x[:, -logits_to_keep:]))
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def _shift_t_value(self, value, device, dtype):
        t = torch.sigmoid(torch.tensor(value, dtype=dtype, device=device))
        shift = self.config.timestep_shift
        return shift * t / (1 + (shift - 1) * t)
