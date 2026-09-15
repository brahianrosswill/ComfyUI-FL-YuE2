"""Dataset manifests and explicit, content-addressed preparation artifacts."""
import hashlib
import json
import os
import re
from pathlib import Path

from filelock import FileLock
import numpy as np
import soundfile as sf
import torch

from ..downloads import digest


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes the selected directory")
    return path


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Run name must use 1–80 letters, digits, underscores or hyphens")
    return value


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_run(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Windows readers must release the file before the worker atomically replaces it.
    with FileLock(str(path) + ".lock"):
        return read_json(path)


def write_run(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        write_json(path, value)


def audio_files(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select an audio directory")
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in {".wav", ".flac", ".mp3"} and p.is_file())
    if not files:
        raise ValueError("No WAV, FLAC or MP3 files in this directory")
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("Audio filenames must have unique stems for sidecars")
    return [contained(root, p.name) for p in files]


def sidecar(audio, kind):
    return Path(audio).with_suffix(f".{kind}.txt")


def dataset(directory, trigger, default_style, validation_fraction, seed, captions=None):
    files = audio_files(directory)
    root = files[0].parent
    songs, hashes, errors = [], set(), []
    for audio in files:
        caption, lyrics = sidecar(audio, "caption"), sidecar(audio, "lyrics")
        metadata = audio.with_suffix(".caption.json")
        if not lyrics.is_file():
            errors.append(f"{audio.name}: missing .lyrics.txt (empty for instrumental)")
            continue
        text = lyrics.read_text(encoding="utf-8").strip()
        style = caption.read_text(encoding="utf-8").strip() if caption.is_file() else default_style.strip()
        if not style:
            errors.append(f"{audio.name}: missing style caption")
            continue
        sha = digest(audio)
        if metadata.is_file():
            meta = read_json(metadata)
            if meta.get("audio_sha256") and meta["audio_sha256"] != sha:
                errors.append(f"{audio.name}: audio changed since captioning; regenerate captions")
                continue
        info = sf.info(audio)
        if info.frames <= 0 or info.channels not in (1, 2):
            errors.append(f"{audio.name}: expected nonempty mono/stereo audio")
            continue
        if sha in hashes:
            errors.append(f"{audio.name}: duplicate recording")
            continue
        hashes.add(sha)
        identity_file = audio.with_suffix(".song.txt")
        identity = identity_file.read_text(encoding="utf-8").strip() if identity_file.exists() else audio.stem
        songs.append({"name": audio.stem, "audio": str(audio), "sha256": sha, "song": identity,
                      "sidecar_hashes": {str(p): digest(p) if p.exists() else None for p in (caption, lyrics, metadata, identity_file)},
                      "style": f"{trigger.strip()}, {style}" if trigger.strip() else style, "lyrics": text,
                      "seconds": info.duration, "sample_rate": info.samplerate, "channels": info.channels,
                      "instrumental": not text})
    if errors:
        raise ValueError("\n".join(errors))
    groups = sorted({s["song"] for s in songs}, key=lambda name: fingerprint([seed, name]))
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction))) if validation_fraction and len(groups) > 1 else 0
    held = set(groups[:count])
    for song in songs:
        song["split"] = "validation" if song["song"] in held else "train"
    result = {"version": 1, "directory": str(root), "songs": songs, "seed": seed}
    result["fingerprint"] = fingerprint(result)
    path = root / "yue2_dataset.json"
    write_json(path, result)
    return str(path)


def check_dataset(manifest):
    for song in manifest["songs"]:
        if digest(Path(song["audio"])) != song["sha256"]:
            raise ValueError(f"{song['name']}: audio changed; rebuild the dataset")
        for name, sha in song.get("sidecar_hashes", {}).items():
            path = Path(name)
            if (digest(path) if path.exists() else None) != sha:
                raise ValueError(f"{song['name']}: sidecars changed; rebuild the dataset")


def load_kit_pack(path):
    # The known kit contains NumPy arrays. Permit only NumPy's array constructors.
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, (np.core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct"), np.ndarray, np.dtype, type(np.dtype(np.int32))]):
        data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty minted regularizer list")
    for item in data:
        codec = np.asarray(item["codec"])
        if codec.ndim != 1 or not np.issubdtype(codec.dtype, np.integer) or not len(codec) or codec.min() < 0 or codec.max() >= 32768:
            raise ValueError("Invalid regularizer semantic tokens")
        if item["src"] not in {"minted", "minted_val"} or not isinstance(item["style"], str) or not isinstance(item["lyrics"], str):
            raise ValueError("Invalid regularizer record")
    return data


def regularizer(path):
    path = Path(path)
    if path.suffix == ".pt":
        return load_kit_pack(path)
    manifest = read_json(path)
    records = []
    for song in manifest["songs"]:
        if not song.get("true_tokens"):
            raise ValueError("Regularizer must contain genuine generated tokens, not predicted labels")
        for key in ("tokens",):
            if not song.get(key) or not Path(song[key]).is_file():
                raise ValueError(f"Generated regularizer is missing {key}")
        records.append({**song, "codec": np.load(song["tokens"], allow_pickle=False),
                        "src": "minted_val" if song["split"] == "validation" else "minted"})
    return records
