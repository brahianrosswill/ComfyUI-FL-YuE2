"""Pinned training assets, downloaded only when a training node is queued."""
import json
from pathlib import Path

from filelock import FileLock
import folder_paths

from ..adapters import import_kit
from ..downloads import contained, digest, transfer_url


TOKENIZER_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
TOKENIZER_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"
ASSETS = {
    "tokenizer_head_joint_v4.pt": (
        f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/tokenizer_head_joint_v4.pt",
        "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e"),
    "nar_lora_joint_v4.pt": (
        f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/nar_lora_joint_v4.pt",
        "df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe"),
    "minted_regularizer_pack.pt": (
        "https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus/resolve/5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35/regularizer/minted_regularizer_pack.pt",
        "bdd9b9780de46bb0752c3e3bc101869759493443c6e35b1b2d4a2c5033eadc4e"),
}


def asset(name, download=True):
    if name not in ASSETS:
        raise ValueError("Unknown training asset")
    candidates = [contained(Path(root) / "training_assets", name) for root in folder_paths.get_folder_paths("yue2")]
    path = next((p for p in candidates if p.is_file()), candidates[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    url, expected = ASSETS[name]
    with FileLock(str(path) + ".lock"):
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {name}; enable download_missing")
            transfer_url(url, path)
        if digest(path) != expected:
            raise ValueError(f"Corrupt training asset: {path.name}; remove it and queue again")
    return path


def acoustic_adapter(model, download=True):
    source = asset("nar_lora_joint_v4.pt", download)
    root = Path(folder_paths.get_folder_paths("loras")[0]) / "YuE2" / "pretrained"
    root.mkdir(parents=True, exist_ok=True)
    target = contained(root, "nar_lora_joint_v4.safetensors")
    with FileLock(str(target) + ".lock"):
        if not target.is_file():
            temporary = target.with_suffix(".tmp")
            import_kit(source, temporary, model / "model.safetensors")
            temporary.replace(target)
    return target


def alignment(download=True):
    roots = [Path(root) / "training_assets" / "alignment" for root in folder_paths.get_folder_paths("yue2")]
    names = ("demucs/955717e8-8726e21a.th", "mms_alignment.pt")
    root = next((p for p in roots if all(contained(p, name).is_file() for name in names)), roots[0])
    root.mkdir(parents=True, exist_ok=True)
    urls = ("https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt")
    with FileLock(str(root / ".download.lock")):
        manifest = {}
        for name, url in zip(names, urls):
            path = contained(root, name)
            if not path.is_file():
                if not download:
                    raise FileNotFoundError("Missing lyric alignment weights; enable download_missing")
                transfer_url(url, path)
            sha = digest(path)
            if name.startswith("demucs/") and not sha.startswith("8726e21a"):
                raise ValueError("Corrupt Demucs alignment weights")
            if name == "mms_alignment.pt" and sha != "20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680":
                raise ValueError("Corrupt MMS alignment weights")
            manifest[path.name] = {"sha256": sha, "bytes": path.stat().st_size, "source": url}
        (root / "demucs/htdemucs.yaml").write_text("models: ['955717e8']\n", encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
