"""Training initialization and portable paired-adapter checkpoints."""
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..adapters import targets, read_adapter
from ..downloads import resolve, digest, MODELS
from .data import read_json, write_json, contained
from .math import LoRALinear
from .prepare import load_head
from .trainer import load_model


FORMAT = "fl-yue2-audio-adapter-v1"


def initialize(assets, mode, rank):
    model = load_model(assets["model"])
    head = load_head(assets["head"], "cuda")
    initial = read_adapter(assets["initial_nar"])[0] if assets.get("initial_nar") else {}
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
    if mode in ("acoustic", "joint", "conditioned"):
        for key in targets(model.config.num_hidden_layers, "nar"):
            parent, name = key.rsplit(".", 1)
            setattr(model.get_submodule(parent), name, LoRALinear(model.get_submodule(key), rank))
        model.vae2llm.float().requires_grad_(True)
        model.llm2vae.float().requires_grad_(True)
    head.requires_grad_(mode in ("head", "joint"))
    head.eval()
    return model, head, initial


def export_bundle(directory, model, head, initial, identity, assets, mode, step, conditioner=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(directory / "head.safetensors"))
    values = {k: v.contiguous() for k, v in initial.items()}
    if mode in ("acoustic", "joint", "conditioned"):
        for key in targets(model.config.num_hidden_layers, "nar"):
            module = model.get_submodule(key)
            a, b = module.A.detach().cpu(), module.B.detach().cpu()
            if key + ".lora_down.weight" in initial:
                a = torch.cat((initial[key + ".lora_down.weight"].float(), a.float()), 0)
                b = torch.cat((initial[key + ".lora_up.weight"].float(), b.float()), 1)
            values[key + ".lora_down.weight"] = a.contiguous()
            values[key + ".lora_up.weight"] = b.contiguous()
        with safe_open(str(Path(assets["model"]) / "model.safetensors"), framework="pt", device="cpu") as base:
            for module in ("vae2llm", "llm2vae"):
                for name, suffix in (("weight", ".diff"), ("bias", ".diff_b")):
                    value = getattr(model.get_submodule(module), name).detach().cpu().float()
                    values[module + suffix] = (value - base.get_tensor(module + "." + name).float()).contiguous()
    if values:
        save_file(values, str(directory / "acoustic.safetensors"), metadata={"format": "fl-yue2-lora-v1", "branch": "nar"})
    manifest = {"format": FORMAT, "mode": mode, "step": step, "base_hash": identity["model"],
                "mert_revision": identity["mert_revision"], "vae_revision": identity["vae_revision"],
                "features": "mert-layer20-25hz-instance-normalized", "head": "head.safetensors",
                "acoustic": "acoustic.safetensors" if values else ""}
    manifest["files"] = {name: digest(directory / name) for name in (manifest["head"], manifest["acoustic"]) if name}
    if conditioner is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in conditioner.state_dict().items()}, str(directory / "conditioning.safetensors"))
        manifest["conditioning"] = {"file": "conditioning.safetensors", "layers": list(conditioner.layers),
                                    "input": "source-vae-posterior-mean-25hz", "semantic_tokens": "source"}
        manifest["files"]["conditioning.safetensors"] = digest(directory / "conditioning.safetensors")
    write_json(directory / "manifest.json", manifest)
    return str(directory / "manifest.json")


def load_bundle(path):
    path = Path(path)
    data = read_json(path)
    if data.get("format") != FORMAT or data.get("features") != "mert-layer20-25hz-instance-normalized":
        raise ValueError("Select a paired-audio adapter manifest")
    if data["mert_revision"] != MODELS["MERT-v2-FullSong"] or data["vae_revision"] != MODELS["YuE2-Vae"]:
        raise ValueError("Audio adapter requires different MERT/VAE revisions")
    required = {name for name in (data["head"], data["acoustic"]) if name}
    if "conditioning" in data:
        condition = data["conditioning"]
        if condition["input"] != "source-vae-posterior-mean-25hz" or condition["semantic_tokens"] != "source" or condition["layers"] != [0, 7, 14, 21]:
            raise ValueError("Unsupported audio conditioning layout")
        required.add(condition["file"])
    elif data["mode"] == "conditioned":
        raise ValueError("Conditioned adapter is missing its source projections")
    if not data["head"] or set(data["files"]) != required:
        raise ValueError("Audio adapter manifest must include hashes for every weight file")
    model = resolve("YuE2-3B", False)
    if digest(model / "model.safetensors") != data["base_hash"]:
        raise ValueError("Audio adapter does not match the loaded YuE2 base weights")
    for name, expected in data["files"].items():
        if digest(contained(path.parent, name)) != expected:
            raise ValueError("Audio adapter file changed or is incomplete")
    head = contained(path.parent, data["head"])
    acoustic = contained(path.parent, data["acoustic"]) if data["acoustic"] else None
    return data, model, head, acoustic
