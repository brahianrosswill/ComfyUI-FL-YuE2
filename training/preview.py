from pathlib import Path

import soundfile as sf

from .. import runtime
from ..adapters import patch_music
from .data import read_run, write_run, fingerprint


def preview(request, emit, cancelled):
    path = Path(request["run"])
    run = read_run(path)
    music, vae = runtime.load_models(False)
    for model in (music.patcher.model, vae.model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    settings = {k: request[k] for k in ("style", "lyrics", "seed", "max_seconds")}
    tag = fingerprint(settings)[:12]
    for checkpoint in run["checkpoints"]:
        cancelled()
        audio_path = path.parent / f"preview-{checkpoint['step']:06d}-{tag}.flac"
        if not audio_path.exists():
            emit({"type": "status", "message": f"Rendering checkpoint {checkpoint['step']}"})
            paths = [checkpoint["adapter"]]
            if checkpoint["branch"] == "ar" and run["assets"].get("initial_nar"):
                paths.append(run["assets"]["initial_nar"])
            patched = patch_music(music, paths)
            plan = runtime.make_plan(patched, request["style"], request["lyrics"], request["seed"], "off", "", 4096)
            latent, truncated, _ = runtime.render(patched, plan, request["max_seconds"], 1, 0.95, 100, 1.2, 1.01, 32)
            audio = runtime.decode(vae, latent, 512)
            sf.write(audio_path, audio["waveform"][0].T.numpy(), 48000, subtype="PCM_24")
            checkpoint["truncated"] = truncated
        checkpoint["preview"] = audio_path.name
        checkpoint["preview_settings"] = settings
        write_run(path, run)
        emit({"type": "checkpoint", **checkpoint, "run": str(path)})
    return str(path)
