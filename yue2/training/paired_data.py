"""Aligned source/target datasets and versioned feature preparation."""
from pathlib import Path

import soundfile as sf

from ..downloads import digest, MODELS
from .data import audio_files, fingerprint


FORMAT = "fl-yue2-pairs-v1"


def make_dataset(source_directory, target_directory, style, validation_fraction, seed):
    sources, targets = audio_files(source_directory), audio_files(target_directory)
    target_map = {p.stem.casefold(): p for p in targets}
    if {p.stem.casefold() for p in sources} != set(target_map):
        raise ValueError("Source and target folders must contain matching, unique filename stems.")
    pairs = []
    for source in sources:
        target = target_map[source.stem.casefold()]
        a, b = sf.info(source), sf.info(target)
        if a.channels not in (1, 2) or b.channels not in (1, 2) or min(a.duration, b.duration) < 1:
            raise ValueError(f"{source.stem}: expected mono/stereo audio at least one second long")
        if abs(a.duration - b.duration) > max(1 / a.samplerate, 1 / b.samplerate):
            raise ValueError(f"{source.stem}: source/target durations differ; align them before training")
        caption, lyrics, group = (target.with_suffix(s) for s in (".caption.txt", ".lyrics.txt", ".song.txt"))
        pairs.append({"name": source.stem, "source": str(source), "target": str(target),
                      "source_hash": digest(source), "target_hash": digest(target), "seconds": a.duration,
                      "style": caption.read_text(encoding="utf-8").strip() if caption.exists() else style,
                      "lyrics": lyrics.read_text(encoding="utf-8").strip() if lyrics.exists() else "",
                      "group": group.read_text(encoding="utf-8").strip() if group.exists() else source.stem})
    groups = sorted({p["group"] for p in pairs}, key=lambda g: fingerprint([seed, g]))
    if len(groups) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Reserve validation data: provide at least two song groups and a fraction between 0 and 1.")
    held = set(groups[:min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))])
    for pair in pairs:
        pair["split"] = "validation" if pair["group"] in held else "train"
    result = {"format": FORMAT, "pairs": pairs}
    result["fingerprint"] = fingerprint(result)
    return result


def check_pairs(data):
    if data.get("format") != FORMAT:
        raise ValueError("Select a prepared paired-audio dataset")
    for pair in data["pairs"]:
        for key in ("source", "target"):
            if digest(Path(pair[key])) != pair[key + "_hash"]:
                raise ValueError(f"{pair['name']}: {key} audio changed; prepare the pairs again")
        for key, expected in pair.get("artifact_hashes", {}).items():
            if digest(Path(pair[key])) != expected:
                raise ValueError(f"{pair['name']}: prepared {key} changed; prepare again")


def asset_identity(assets):
    return {"model": digest(Path(assets["model"]) / "model.safetensors"),
            "head": digest(Path(assets["head"])),
            "nar": digest(Path(assets["initial_nar"])) if assets.get("initial_nar") else "",
            "mert_revision": assets["mert_revision"], "vae_revision": MODELS["YuE2-Vae"]}

