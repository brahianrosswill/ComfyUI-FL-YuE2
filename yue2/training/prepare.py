import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from scipy.signal import resample_poly
from transformers import AutoModel, AutoFeatureExtractor

from .data import read_json, write_json, fingerprint, check_dataset
from ..downloads import digest
from .math import TokenHead


def load_head(path, device):
    state = load_file(str(path)) if Path(path).suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)["model"]
    with torch.device("meta"):
        head = TokenHead()
    head.load_state_dict(state, strict=True, assign=True)
    return head.to(device).eval()


def normalize(features):
    features = features.astype(np.float32)
    return (features - features.mean(0)) / (features.std(0) + 1e-5)


def predict(head, features, cancelled, input_normalized=False):
    features = features.astype(np.float32) if input_normalized else normalize(features)
    total, window = len(features), head.pos.shape[1]
    out = np.zeros(total, dtype=np.int32)
    starts = list(range(0, max(1, total - window + 1), window // 2))
    if starts[-1] + window < total:
        starts.append(max(0, total - window))
    device = next(head.parameters()).device
    for start in starts:
        cancelled()
        value = features[start:start + window]
        count = len(value)
        value = np.pad(value, ((0, window - count), (0, 0)))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids = head(torch.tensor(value[None], device=device))[0, :count].detach().float().argmax(-1).cpu().numpy()
        lo, hi = start + (0 if start == 0 else window // 4), start + count - (0 if start + count >= total else window // 4)
        out[lo:hi] = ids[lo - start:hi - start]
    return out


def resample(audio, source, target):
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor, axis=0).astype(np.float32) if source != target else audio


def mert_features(model, processor, mono, cancelled):
    device = next(model.parameters()).device
    chunks = [mono[start:start + 720000] for start in range(0, len(mono), 720000)]
    chunks = [chunk for chunk in chunks if len(chunk) >= 24000]
    if not chunks:
        raise ValueError("Training audio must be at least one second")
    features = []
    for chunk in chunks:
        cancelled()
        inputs = {k: v.to(device) for k, v in processor([chunk], sampling_rate=24000, return_tensors="pt").items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(**inputs, output_hidden_states=True).hidden_states[20][0]
        features.append(value.detach().float().cpu())
    joined = torch.cat(features)
    return F.interpolate(joined.T[None], size=round(len(mono) / 24000 * 25), mode="linear", align_corners=False)[0].T.half().numpy()


def prepare(request, emit, cancelled):
    source = read_json(request["dataset"])
    check_dataset(source)
    assets = request["assets"]
    root = Path(request["cache_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    head_hash = digest(Path(assets["head"]))
    feature_version = fingerprint([assets["mert_revision"], "layer20-kit30s-v1"])
    result = {**source, "head_hash": head_hash, "mode": "ar", "songs": []}
    # Preparation has no backward path; parameters do not require gradients.
    mert = processor = head = None
    try:
        for index, song in enumerate(source["songs"]):
            cancelled()
            emit({"type": "status", "message": f"Preparing {index + 1}/{len(source['songs'])}: {song['name']}"})
            folder = root / song["sha256"]
            folder.mkdir(exist_ok=True)
            feature_path = folder / f"features-{feature_version[:16]}.npy"
            tokens_path = folder / f"tokens-{feature_version[:16]}-{head_hash[:16]}.npy"
            audio, sr = sf.read(song["audio"], dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"{song['name']}: non-finite audio")
            if not feature_path.exists():
                if mert is None:
                    processor = AutoFeatureExtractor.from_pretrained(assets["mert"], local_files_only=True)
                    mert = AutoModel.from_pretrained(assets["mert"], trust_remote_code=True, local_files_only=True).to("cuda").eval()
                    for parameter in mert.parameters():
                        parameter.requires_grad_(False)
                features = mert_features(mert, processor, resample(audio.mean(1), sr, 24000), cancelled)
                np.save(feature_path, features)
            else:
                features = np.load(feature_path, allow_pickle=False)
            if not tokens_path.exists():
                if head is None:
                    head = load_head(assets["head"], "cuda")
                    for parameter in head.parameters():
                        parameter.requires_grad_(False)
                np.save(tokens_path, predict(head, features, cancelled))
            row = {**song, "features": str(feature_path), "tokens": str(tokens_path)}
            if request.get("align") and song["lyrics"]:
                from .alignment import align
                row["cursor"] = align(song, folder, assets, emit, cancelled)
            result["songs"].append(row)
            emit({"type": "progress", "step": index + 1, "max_steps": len(source["songs"])})
    finally:
        del mert, processor, head
    result["fingerprint"] = fingerprint(result)
    path = root / f"prepared-{result['fingerprint'][:16]}.json"
    write_json(path, result)
    return str(path)
