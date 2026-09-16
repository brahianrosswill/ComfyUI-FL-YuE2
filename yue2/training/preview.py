from pathlib import Path
import os
import time

import soundfile as sf

from .. import runtime
from ..adapters import patch_music
from .data import read_run, write_run, fingerprint


def preview(request, emit, cancelled):
    path = Path(request["run"])
    run = read_run(path)
    emit({"type": "preview_progress", "phase": "loading", "step": 0, "done": 0, "total": 0})
    music, vae = runtime.load_models(False, model_directory=run["assets"]["model"])
    for model in (music.patcher.model, vae.model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    settings = {k: request[k] for k in ("style", "lyrics", "seed", "max_seconds")}
    settings.update(ar_strength=request.get("ar_strength", 1.0), nar_strength=request.get("nar_strength", 1.0))
    planning = run.get("planning", "off")
    settings["planning"] = planning
    tag = fingerprint([settings, run["signature"]])[:12]
    baseline = run.setdefault("baseline", {"step": 0})
    for checkpoint in [baseline, *run["checkpoints"]]:
        cancelled()
        step = checkpoint["step"]
        audio_path = path.parent / f"preview-{step:06d}-{tag}.flac"
        last_phase, last_emit = None, 0

        def progress(phase, done=0, total=0):
            nonlocal last_phase, last_emit
            cancelled()
            now = time.monotonic()
            if phase != last_phase or done == total or now - last_emit >= 0.2:
                emit({"type": "preview_progress", "phase": phase, "step": step, "done": done, "total": total})
                last_phase, last_emit = phase, now

        if not audio_path.exists():
            progress("loading")
            paths = [] if step == 0 else [checkpoint["adapter"]]
            starting_nar = run["assets"].get("initial_nar", "")
            if run.get("recipe") == "ai_toolkit_joint_v1" and run["config"].get("nar_start") == "base":
                starting_nar = ""
            acoustic = checkpoint.get("acoustic_adapter", starting_nar)
            if acoustic:
                paths.append(acoustic)
            patched = patch_music(music, paths, settings["ar_strength"], settings["nar_strength"]) if paths else music
            plan = runtime.make_plan(patched, request["style"], request["lyrics"], request["seed"], planning, "", 4096)
            progress("tokens", 0, round(request["max_seconds"] * 25))
            latent, truncated, _ = runtime.render(patched, plan, request["max_seconds"], 1, 0.95, 100, 1.2, 1.01 if planning == "off" else 1.0, 32,
                                                  on_progress=progress, check_cancelled=cancelled)
            audio = runtime.decode(vae, latent, 512, on_progress=progress, check_cancelled=cancelled)
            progress("saving")
            temporary = audio_path.with_suffix(".tmp")
            try:
                sf.write(temporary, audio["waveform"][0].T.numpy(), 48000, format="FLAC", subtype="PCM_24")
                cancelled()
                os.replace(temporary, audio_path)
            finally:
                temporary.unlink(missing_ok=True)
            checkpoint["truncated"] = truncated
        checkpoint["preview"] = audio_path.name
        checkpoint["preview_settings"] = settings
        write_run(path, run)
        progress("complete", 1, 1)
        emit({"type": "checkpoint", **checkpoint, "run": str(path)})
    return str(path)
