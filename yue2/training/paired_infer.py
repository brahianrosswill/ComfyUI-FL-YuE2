"""Audio-conditioned inference and source/target checkpoint comparisons."""
from pathlib import Path
import os
import time

import numpy as np
import soundfile as sf
import torch
import comfy.ops
from safetensors.torch import load_file
from transformers import AutoModel, AutoFeatureExtractor

from .. import runtime
from ..adapters import patch_music
from ..downloads import resolve
from ..nar import synthesize
from ..conditioning import AudioConditioner
from .data import read_json, read_run, write_run, fingerprint, contained
from .paired_prepare import read_audio, extract_features, load_encoder, encode_target
from .paired_models import load_bundle
from .prepare import load_head, predict, normalize


def render_features(music, vae, features, bundle, style, lyrics, seed, steps, progress, cancelled, source_latents=None, condition_scale=1.0):
    manifest, _, head_path, acoustic = load_bundle(bundle)
    progress("tokens", 0, len(features))
    head = load_head(head_path, "cuda").requires_grad_(False)
    with torch.set_grad_enabled(False):
        tokens = predict(head, features, cancelled, input_normalized=True)
    del head
    progress("tokens", len(tokens), len(tokens))
    patched = patch_music(music, [str(acoustic)]) if acoustic else music
    plan = runtime.make_plan(patched, style, lyrics, seed, "off", "", 4096)
    model = patched.prepare(len(plan.prefix) + min(len(tokens), 8192) * 2 + 4)
    conditioner = None
    if "conditioning" in manifest:
        if source_latents is None:
            raise ValueError("This adapter requires source VAE latents")
        metadata = manifest["conditioning"]
        conditioner = AudioConditioner(model.config.hidden_size, metadata["layers"], comfy.ops.manual_cast)
        conditioner.load_state_dict(load_file(str(contained(Path(bundle).parent, metadata["file"]))))
        conditioner.to(device=model.vae2llm.weight.device, dtype=model.vae2llm.weight.dtype)
        source_latents = torch.as_tensor(source_latents)
    with torch.set_grad_enabled(False):
        latent = synthesize(model, plan.prefix, tokens.tolist(), seed, steps=steps, cancelled=cancelled,
                            on_progress=lambda done, total: progress("synthesis", done, total),
                            conditioner=conditioner, condition=source_latents, condition_scale=condition_scale)
        return runtime.decode(vae, latent.T[None].contiguous(), 512, on_progress=progress, check_cancelled=cancelled)


def write_audio(path, audio):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        sf.write(temporary, audio["waveform"][0].T.numpy(), audio["sample_rate"], format="FLAC", subtype="PCM_24")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def infer(request, emit, cancelled):
    manifest, _, _, _ = load_bundle(request["bundle"])
    cancelled()
    emit({"type": "status", "message": "Extracting source audio features"})
    audio = read_audio(request["source"])
    if len(audio) < 48000:
        raise ValueError("Source audio must be at least one second")
    frames = (len(audio) + 1919) // 1920
    audio = np.pad(audio, ((0, frames * 1920 - len(audio)), (0, 0)))
    root = resolve("MERT-v2-FullSong", False)
    processor = AutoFeatureExtractor.from_pretrained(root, local_files_only=True)
    mert = AutoModel.from_pretrained(root, trust_remote_code=True, local_files_only=True).cuda().eval().requires_grad_(False)
    with torch.set_grad_enabled(False):
        features = normalize(extract_features(mert, processor, audio, cancelled))
    del mert, processor
    source_latents = None
    if "conditioning" in manifest:
        emit({"type": "status", "message": "Encoding source audio conditioning"})
        encoder = load_encoder()
        with torch.set_grad_enabled(False):
            source_latents = encode_target(encoder, audio, cancelled)
        del encoder
    music, vae = runtime.load_models(False)
    def progress(phase, done, total):
        cancelled()
        emit({"type": "preview_progress", "phase": phase, "step": 0, "done": done, "total": total})
    output = render_features(music, vae, features, request["bundle"], request["style"], request["lyrics"], request["seed"], request["sampling_steps"], progress, cancelled, source_latents, request.get("condition_scale", 1.0))
    write_audio(request["output"], output)
    return request["output"]


def preview_pairs(request, emit, cancelled):
    path = Path(request["run"])
    run = read_run(path)
    data = read_json(run["dataset"])
    held = [p for p in data["pairs"] if p["split"] == "validation"]
    index = request["validation_index"]
    if not 0 <= index < len(held):
        raise ValueError(f"Validation index must be between 0 and {len(held) - 1}")
    pair = held[index]
    features = normalize(np.load(pair["features"], allow_pickle=False))[:round(request["max_seconds"] * 25)]
    settings = {k: request[k] for k in ("seed", "max_seconds", "sampling_steps", "validation_index")}
    source_latents = None
    if run["config"]["mode"] == "conditioned":
        settings["condition_scale"] = request.get("condition_scale", 1.0)
        source_latents = np.load(pair["source_latents"], allow_pickle=False)[:len(features)]
    tag = fingerprint([settings, run["signature"]])[:12]
    run["references"] = []
    for step, key in ((-2, "source"), (-1, "target")):
        name = f"reference-{key}-{tag}.flac"
        audio = read_audio(pair[key])[:len(features) * 1920]
        sf.write(path.parent / name, audio, 48000, subtype="PCM_24")
        run["references"].append({"step": step, "label": key.title(), "preview": name})
    write_run(path, run)
    emit({"type": "checkpoint", "run": str(path)})
    emit({"type": "preview_progress", "phase": "loading", "step": 0, "done": 0, "total": 0})
    music, vae = runtime.load_models(False)
    for item in [run["baseline"], *run["checkpoints"]]:
        cancelled()
        step = item["step"]
        output = path.parent / f"preview-{step:06d}-{tag}.flac"
        last = 0
        def progress(phase, done, total):
            nonlocal last
            cancelled()
            now = time.monotonic()
            if now - last >= .2 or done == total:
                emit({"type": "preview_progress", "phase": phase, "step": step, "done": done, "total": total})
                last = now
        if not output.exists():
            audio = render_features(music, vae, features, item["bundle"], pair["style"], pair["lyrics"], settings["seed"], settings["sampling_steps"], progress, cancelled, source_latents, settings.get("condition_scale", 1.0))
            progress("saving", 0, 0)
            write_audio(output, audio)
        item.update({"preview": output.name, "preview_settings": settings})
        write_run(path, run)
        emit({"type": "checkpoint", "step": step, "run": str(path)})
    return str(path)
