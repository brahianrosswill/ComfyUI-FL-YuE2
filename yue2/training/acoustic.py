"""Shared acoustic LoRA initialization and export for both trainers."""
from pathlib import Path
import random

import numpy as np
import torch
from safetensors import safe_open

from ..adapters import targets, read_adapter
from .math import LoRALinear, flow_loss
from .paired_prepare import read_audio, load_encoder, encode_target, save_array
from .data import fingerprint
from ..downloads import MODELS
from ..protocol import CODEC_OFFSET


def fold_acoustic(model, path):
    initial = read_adapter(path)[0] if path else {}
    with torch.set_grad_enabled(False):
        for key in targets(model.config.num_hidden_layers, "nar"):
            module = model.get_submodule(key)
            if key + ".lora_down.weight" in initial:
                delta = initial[key + ".lora_up.weight"].float() @ initial[key + ".lora_down.weight"].float()
                module.weight.add_(delta.to(module.weight))
        for module in ("vae2llm", "llm2vae"):
            for name, suffix in (("weight", ".diff"), ("bias", ".diff_b")):
                parameter = getattr(model.get_submodule(module), name)
                if module + suffix in initial:
                    parameter.add_(initial[module + suffix].to(parameter))
    return initial


def install_acoustic(model, rank):
    for key in targets(model.config.num_hidden_layers, "nar"):
        parent, name = key.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, LoRALinear(model.get_submodule(key), rank))
    model.vae2llm.float().requires_grad_(True)
    model.llm2vae.float().requires_grad_(True)


def acoustic_weights(model, initial, model_directory, trained=True):
    values = {k: v.contiguous() for k, v in initial.items()}
    if trained:
        for key in targets(model.config.num_hidden_layers, "nar"):
            module = model.get_submodule(key)
            a, b = module.A.detach().cpu(), module.B.detach().cpu()
            if key + ".lora_down.weight" in initial:
                a = torch.cat((initial[key + ".lora_down.weight"].float(), a.float()), 0)
                b = torch.cat((initial[key + ".lora_up.weight"].float(), b.float()), 1)
            values[key + ".lora_down.weight"] = a.contiguous()
            values[key + ".lora_up.weight"] = b.contiguous()
        with safe_open(str(Path(model_directory) / "model.safetensors"), framework="pt", device="cpu") as base:
            for module in ("vae2llm", "llm2vae"):
                for name, suffix in (("weight", ".diff"), ("bias", ".diff_b")):
                    value = getattr(model.get_submodule(module), name).detach().cpu().float()
                    values[module + suffix] = (value - base.get_tensor(module + "." + name).float()).contiguous()
    return values


def prepare_targets(songs, directory, emit, cancelled):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    encoder = None
    try:
        for index, item in enumerate(songs):
            cancelled()
            frames = len(item["codec"])
            tag = fingerprint([item["sha256"], frames, MODELS["YuE2-Vae"], "posterior-mean-512-halo64-v1"])
            path = directory / (tag + ".npy")
            if not path.exists():
                emit({"type": "status", "message": f"Encoding acoustic target {index + 1}/{len(songs)}"})
                if encoder is None:
                    encoder = load_encoder()
                audio = read_audio(item["audio"])[:frames * 1920]
                audio = np.pad(audio, ((0, frames * 1920 - len(audio)), (0, 0)))
                with torch.set_grad_enabled(False):
                    latents = encode_target(encoder, audio, cancelled)
                save_array(path, latents)
            item["latents"] = np.load(path, allow_pickle=False)
            if item["latents"].shape != (frames, 64):
                raise ValueError(f"{item['name']}: acoustic target/token alignment failed")
    finally:
        del encoder


def acoustic_loss(model, item, seed, training=True, t=None):
    device = model.model.embed_tokens.weight.device
    rng = random.Random(seed)
    frames = min(512, len(item["codec"]))
    offset = rng.randrange(len(item["codec"]) - frames + 1) if training else (len(item["codec"]) - frames) // 2
    target = torch.from_numpy(item["latents"][offset:offset + frames].copy()).to(device)
    ids = torch.as_tensor(item["codec"][offset:offset + frames].astype(np.int64), device=device) + CODEC_OFFSET
    embeddings = model.model.embed_tokens(ids)
    noise = torch.randn(target.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    t = min(.98, max(.02, rng.betavariate(2, 2))) if t is None else t
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return flow_loss(model, item["prefix"], embeddings, target, t, noise, False, training)
