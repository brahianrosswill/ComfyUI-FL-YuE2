"""Training initialization and portable paired-adapter checkpoints."""
from pathlib import Path

from safetensors.torch import save_file

from ..downloads import resolve, digest, MODELS
from .data import read_json, write_json, contained
from .acoustic import fold_acoustic, install_acoustic, acoustic_weights
from .prepare import load_head
from .trainer import load_model


FORMAT = "fl-yue2-audio-adapter-v1"


def initialize(assets, mode, rank):
    model = load_model(assets["model"])
    head = load_head(assets["head"], "cuda")
    initial = fold_acoustic(model, assets.get("initial_nar", ""))
    if mode in ("acoustic", "joint", "conditioned"):
        install_acoustic(model, rank)
    head.requires_grad_(mode in ("head", "joint"))
    head.eval()
    return model, head, initial


def export_bundle(directory, model, head, initial, identity, assets, mode, step, conditioner=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(directory / "head.safetensors"))
    values = acoustic_weights(model, initial, assets["model"] if mode != "head" else None, mode in ("acoustic", "joint", "conditioned"))
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
