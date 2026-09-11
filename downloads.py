import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from filelock import FileLock
import folder_paths
from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted


MODELS = {
    "YuE2-3B": "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
    "YuE2-Vae": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
}
COMMON_FILES = ("config.json", "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md",
                "licenses/SnakeBeta-NVIDIA-MIT.txt", "licenses/stable-audio-tools-MIT.txt")
folder_paths.add_model_folder_path("yue2", str(Path(folder_paths.models_dir) / "yue2"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            throw_exception_if_processing_interrupted()
            result.update(block)
    return result.hexdigest()


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("YuE2 model file escapes its model directory")
    return path


def transfer(name, revision, filename, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError("YuE2 partial download must not be a symlink")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(f"https://huggingface.co/m-a-p/{name}/resolve/{revision}/{filename}",
                      headers={"Range": f"bytes={offset}-"} if offset else {})
    logging.info("YuE2: downloading %s/%s", name, filename)
    with urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Unexpected download resume offset")
        completed = offset if resumed else 0
        total = completed + int(response.headers.get("Content-Length", 0))
        progress = ProgressBar(max(total, 1))
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                throw_exception_if_processing_interrupted()
                stream.write(block)
                completed += len(block)
                progress.update_absolute(completed, max(total, completed))
        if total and completed != total:
            raise IOError(f"Incomplete YuE2 download: {filename}; queue again to resume")
    os.replace(partial, target)


def resolve(name, download=True):
    if name not in MODELS:
        raise ValueError("Unknown YuE2 model selection")
    revision = MODELS[name]
    files = list(COMMON_FILES)
    if name == "YuE2-3B":
        files += ["qwen.tiktoken", "generation_config.json", "yue2_generation_config.json"]
    roots = folder_paths.get_folder_paths("yue2")
    candidates = [contained(root, name) for root in roots]
    target = next((path for path in candidates if all(contained(path, file).is_file() for file in files + ["model.safetensors"])),
                  next((path for path in candidates if path.is_dir()), candidates[0]))
    if not download and not target.is_dir():
        raise FileNotFoundError(f"Install {name} in {target} or enable download_missing")
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(contained(target, ".download.lock"))):
        for filename in files + ["model.safetensors"]:
            path = contained(target, filename)
            if not path.is_file():
                if not download:
                    raise FileNotFoundError(f"Missing {path}; enable download_missing to complete installation")
                transfer(name, revision, filename, path)
        manifest = json.loads((target / "weights_manifest.json").read_text())
        if set(manifest["files"]) != {"model.safetensors"}:
            raise ValueError("Unexpected YuE2 weight manifest")
        weight = contained(target, "model.safetensors")
        stamp = contained(target, ".verified.json")
        state = {"revision": revision, "manifest": digest(target / "weights_manifest.json"),
                 "files": {filename: [contained(target, filename).stat().st_size,
                                      contained(target, filename).stat().st_mtime_ns]
                           for filename in files + ["model.safetensors"]}}
        previous = json.loads(stamp.read_text()) if stamp.is_file() else None
        if previous != state:
            expected = manifest["files"]["model.safetensors"]
            if weight.stat().st_size != expected["bytes"] or digest(weight) != expected["sha256"]:
                raise ValueError(f"Corrupt YuE2 weights: {weight}. Remove that file and queue again to download it.")
            temporary = contained(target, ".verified.tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            os.replace(temporary, stamp)
    return target
