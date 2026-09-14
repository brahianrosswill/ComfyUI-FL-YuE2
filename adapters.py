"""Translate YuE2 training artifacts at the model-patcher boundary."""

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import comfy.lora

from .runtime import MusicModel


def targets(layers, branch):
    for index in range(layers):
        for module, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                yield f"model.layers.{index}.{'nar_' if branch == 'nar' else ''}{module}.{name}"


def import_kit(source, destination, base_path, layers=28):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    branch = "nar" if "io" in checkpoint else "ar"
    keys = list(targets(layers, branch))
    values, rank = checkpoint["lora"], checkpoint["rank"]
    if len(values) != len(keys) * 2:
        raise ValueError("YuE2 adapter tensor count does not match the base model")
    result = {}
    with safe_open(str(base_path), framework="pt", device="cpu") as base:
        for index, key in enumerate(keys):
            a, b = values[index * 2:index * 2 + 2]
            shape = base.get_slice(key + ".weight").get_shape()
            if list(a.shape) != [rank, shape[1]] or list(b.shape) != [shape[0], rank]:
                raise ValueError(f"Adapter shape mismatch: {key}")
            result[key + ".lora_down.weight"] = a.contiguous()
            result[key + ".lora_up.weight"] = b.contiguous()
        if branch == "nar":
            for module in ("vae2llm", "llm2vae"):
                for kind in ("weight", "bias"):
                    key = f"{module}.{kind}"
                    value = checkpoint["io"][module][kind].float()
                    original = base.get_tensor(key).float()
                    if value.shape != original.shape:
                        raise ValueError(f"Projection shape mismatch: {key}")
                    result[module + (".diff" if kind == "weight" else ".diff_b")] = value - original
    save_file(result, str(destination), metadata={"format": "fl-yue2-lora-v1", "branch": branch, "rank": str(rank)})
    return str(destination)


def read_adapter(path):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        meta = file.metadata() or {}
    if meta.get("format") != "fl-yue2-lora-v1" or meta.get("branch") not in {"ar", "nar"}:
        raise ValueError("Select a converted FL YuE2 adapter")
    return load_file(str(path)), meta


def patch_music(music, paths, ar_strength=1.0, nar_strength=1.0):
    patcher = music.patcher.clone()
    state = patcher.model.state_dict()
    for path in paths:
        if not path:
            continue
        tensors, metadata = read_adapter(path)
        branch = metadata["branch"]
        keys = list(targets(patcher.model.config.num_hidden_layers, branch))
        if branch == "nar":
            keys += ["vae2llm", "llm2vae"]
        mapping = {key: key + ".weight" for key in keys}
        allowed = {key + suffix for key in keys for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha", ".diff", ".diff_b")}
        if set(tensors) - allowed:
            raise ValueError("Adapter contains unsupported tensors")
        for key in keys:
            down, up = tensors.get(key + ".lora_down.weight"), tensors.get(key + ".lora_up.weight")
            if (down is None) != (up is None):
                raise ValueError(f"Incomplete adapter pair: {key}")
            if down is not None:
                shape = state[key + ".weight"].shape
                if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or (up.shape[0], down.shape[1]) != tuple(shape):
                    raise ValueError(f"Adapter shape mismatch: {key}")
            for suffix, kind in ((".diff", ".weight"), (".diff_b", ".bias")):
                if key + suffix in tensors and tensors[key + suffix].shape != state[key + kind].shape:
                    raise ValueError(f"Projection shape mismatch: {key}")
        patches = comfy.lora.load_lora(tensors, mapping, log_missing=False)
        if not patches:
            raise ValueError("Adapter has no applicable tensors")
        accepted = patcher.add_patches(patches, ar_strength if branch == "ar" else nar_strength)
        if set(accepted) != set(patches):
            raise ValueError("Adapter does not match this YuE2 model")
    return MusicModel(patcher, music.tokenizer, music.generation)
