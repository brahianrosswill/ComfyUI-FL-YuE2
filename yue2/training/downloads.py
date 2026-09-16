"""Pinned training assets, downloaded only when a training node is queued."""
import json
from pathlib import Path

from filelock import FileLock
import folder_paths
from huggingface_hub import snapshot_download

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

SHEETSAGE_REVISION = "2c45f222a05d90783d7a05ea8503ad1b5408bac6"
SHEETSAGE_MODEL_REVISION = "8e6fcf0f23252ed188b634bd50d44f4b01fba890"
SHEETSAGE_WEIGHT = (
    f"https://huggingface.co/Comfy-Org/YuE2/resolve/{SHEETSAGE_MODEL_REVISION}/audio_encoders/sheetsage2_bf16.safetensors",
    "5fd960ce3df281e3f3a889d174584d88f96247711480cf96377b12d7e8b6adc5",
)
SHEETSAGE_SOURCE_HASHES = {
    "config.json": "a986e63f5d831ecb823c11d19cfb371d763f25ae2e614ec9acd714c0b8bb87fd",
    "modeling_sheetsage2.py": "df36bec866e4bd1f0a08e79369405c9a1bd86214a1c48559e75e3e567f43bbc4",
    "pipeline_sheetsage2.py": "0ae572554ca375323e2aa4ea1fb25ba27e7f4d520c0ba5b101d74930c0a66c33",
    "notation_sheetsage2.py": "f75f0e8b268cd9e9c622d791d3fd77d2e68ad355a4e67018c7e15aeb2bc05af7",
    "LICENSE": "73593d6cad4ca80f3ab1448a4908f92ab802dfffe029561e3b12fa7d07c8b87c",
}
SHEETSAGE_COMPAT = {
    "tokenizer.py": "1e128203033c30018c1a8b95beecf58a1c4869b5b473b41ecc98bd54823cef22",
    "sheetsage_decoder.py": "b7e44bb4bc8c2df3c0d9262e1529ab8e65514c816bdbdfe7862293af4c25d721",
    "sheetsage_generate.py": "dce23bf27e3a0f8c1fa5f34e5f09af6f69736034a945eafeacea473d430df607",
    "sheetsage_repair.py": "b65945fe219f0b68b55e96e8e7d5e817e9655baffb7f8b7878359e85a9f660ab",
    "shims/__init__.py": "50ccd28ef3d733e77506776563f156aa948a726e40debc6508521289da0d1fae",
    "shims/mir_eval_chord.py": "b22517cf0b382d9e6bf3e3bdd04b198c5c3a2d189dab4c6d0e9b8255ed33cb1d",
    "shims/pretty_midi.py": "5b39bde344447e3a82e4ad32be520368fe74048f7bad712e47fd74b980479595",
}
AI_TOOLKIT_REVISION = "410206e41c7a323c41b0eff67fc5d1e02d26c97b"


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


def sheetsage(download=True):
    weight = Path(folder_paths.models_dir) / "audio_encoders" / "sheetsage2_bf16.safetensors"
    weight.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(weight) + ".lock"):
        if not weight.is_file():
            if not download:
                raise FileNotFoundError("Missing SheetSage2 weights; enable download_missing")
            transfer_url(SHEETSAGE_WEIGHT[0], weight)
        if digest(weight) != SHEETSAGE_WEIGHT[1]:
            raise ValueError("Corrupt SheetSage2 weights; remove them and queue again")
    root = Path(folder_paths.get_folder_paths("yue2")[0]) / "training_assets"
    source = root / "sheetsage2-code"
    valid_source = all((source / name).is_file() and digest(source / name) == sha for name, sha in SHEETSAGE_SOURCE_HASHES.items())
    if not valid_source:
        if not download:
            raise FileNotFoundError("Missing pinned SheetSage2 code; enable download_missing")
        snapshot_download("m-a-p/SheetSage2", revision=SHEETSAGE_REVISION, local_dir=source,
                          allow_patterns=["*.py", "config.json", "LICENSE", "THIRD_PARTY_NOTICES.md"], force_download=True)
        if not all((source / name).is_file() and digest(source / name) == sha for name, sha in SHEETSAGE_SOURCE_HASHES.items()):
            raise ValueError("Pinned SheetSage2 source verification failed")
    compat = root / "sheetsage2-compat"
    for name, expected in SHEETSAGE_COMPAT.items():
        path = contained(compat, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if not path.is_file():
                if not download:
                    raise FileNotFoundError("Missing pinned SheetSage2 compatibility code; enable download_missing")
                transfer_url(f"https://raw.githubusercontent.com/ostris/ai-toolkit/{AI_TOOLKIT_REVISION}/extensions_built_in/audio_models/yue2/src/{name}", path)
            if digest(path) != expected:
                raise ValueError(f"Corrupt SheetSage2 compatibility file: {name}")
    return {"weights": str(weight), "source": str(source), "compat": str(compat),
            "revision": SHEETSAGE_REVISION, "compat_revision": AI_TOOLKIT_REVISION}


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
