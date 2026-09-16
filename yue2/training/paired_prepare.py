"""Source MERT features and aligned deterministic VAE latents."""
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from comfy.ldm.audio.autoencoder import OobleckEncoder
from safetensors.torch import load_file
from transformers import AutoModel, AutoFeatureExtractor

from ..downloads import resolve, digest
from .data import read_json, write_json, fingerprint
from .paired_data import check_pairs, asset_identity
from .prepare import resample, mert_features, load_head, predict


def read_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {Path(path).name}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return resample(audio, rate, 48000)


def extract_features(model, processor, audio, cancelled):
    mono = resample(audio.mean(1), 48000, 24000)
    output = []
    for start in range(0, len(mono), 720000):
        cancelled()
        chunk = mono[start:start + 720000]
        frames = round(len(chunk) / 960)
        if not frames:
            continue
        value = np.pad(chunk, (0, max(0, 24000 - len(chunk))))
        output.append(mert_features(model, processor, value, cancelled)[:frames])
    return np.concatenate(output)


def load_encoder():
    root = resolve("YuE2-Vae", False)
    config = read_json(root / "config.json")
    encoder = OobleckEncoder(**config["encoder_config"])
    state = load_file(str(root / "model.safetensors"))
    state = {k[len("encoder."):].replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1"): v
             for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)
    return encoder.cuda().eval().requires_grad_(False)


def encode_target(encoder, audio, cancelled):
    frames = len(audio) // 1920
    output = []
    for start in range(0, frames, 512):
        cancelled()
        end = min(frames, start + 512)
        left, right = max(0, start - 64), min(frames, end + 64)
        value = torch.from_numpy(audio[left * 1920:right * 1920].T.copy())[None].cuda()
        latent = encoder(value).chunk(2, dim=1)[0]
        output.append(latent[0, :, start - left:end - left].T.detach().cpu().float().numpy().copy())
    return np.concatenate(output)


def save_array(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.save(file, value, allow_pickle=False)
    os.replace(temporary, path)


def prepare_pairs(request, emit, cancelled):
    data = read_json(request["dataset"])
    check_pairs(data)
    assets = request["assets"]
    identity = asset_identity(assets)
    version = fingerprint([identity, "paired-mert30s-vae512-halo64-v1"])
    root = Path(request["cache_directory"])
    root.mkdir(parents=True, exist_ok=True)
    mert = processor = encoder = head = None
    try:
        for index, pair in enumerate(data["pairs"]):
            cancelled()
            emit({"type": "status", "message": f"Preparing audio pair {index + 1}/{len(data['pairs'])}: {pair['name']}"})
            folder = root / fingerprint([pair["source_hash"], pair["target_hash"], version])[:24]
            folder.mkdir(exist_ok=True)
            paths = {key: folder / (key + ".npy") for key in ("features", "latents", "source_latents", "anchor_tokens")}
            if not all(p.exists() for p in paths.values()):
                if mert is None:
                    processor = AutoFeatureExtractor.from_pretrained(assets["mert"], local_files_only=True)
                    mert = AutoModel.from_pretrained(assets["mert"], trust_remote_code=True, local_files_only=True).cuda().eval().requires_grad_(False)
                    encoder = load_encoder()
                    head = load_head(assets["head"], "cuda").requires_grad_(False)
                source, target = read_audio(pair["source"]), read_audio(pair["target"])
                frames = (max(len(source), len(target)) + 1919) // 1920
                source = np.pad(source, ((0, frames * 1920 - len(source)), (0, 0)))
                target = np.pad(target, ((0, frames * 1920 - len(target)), (0, 0)))
                with torch.set_grad_enabled(False):
                    features = extract_features(mert, processor, source, cancelled)
                    latents = encode_target(encoder, target, cancelled)
                    source_latents = encode_target(encoder, source, cancelled)
                    tokens = predict(head, features, cancelled)
                if len(features) != frames or len(latents) != frames or len(source_latents) != frames:
                    raise ValueError(f"{pair['name']}: feature/latent frame alignment failed")
                for key, value in (("features", features), ("latents", latents), ("source_latents", source_latents), ("anchor_tokens", tokens)):
                    save_array(paths[key], value)
            pair.update({key: str(path) for key, path in paths.items()})
            pair["artifact_hashes"] = {key: digest(path) for key, path in paths.items()}
            emit({"type": "progress", "step": index + 1, "max_steps": len(data["pairs"])})
    finally:
        del mert, encoder, head, processor
    data["identity"] = identity
    data["fingerprint"] = fingerprint([data["pairs"], version])
    path = root / f"paired-{data['fingerprint'][:16]}.json"
    write_json(path, data)
    return str(path)
