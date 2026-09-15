import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from ..model import YuE2Model
from ..protocol import SongRequest, token_prefixes, CODEC_OFFSET, MUSIC_END
from ..tokenizer import YuE2TextTokenizer
from ..adapters import targets
from .data import read_json, read_run, write_run, fingerprint, regularizer, check_dataset
from .math import LoRALinear, ar_loss
from ..downloads import digest


def load_model(path, device="cuda"):
    with torch.device("meta"):
        model = YuE2Model(read_json(Path(path) / "config.json"), operations=nn)
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")), strict=True, assign=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def install_lora(model, rank):
    for key in targets(model.config.num_hidden_layers, "ar"):
        parent_name, name = key.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, name, LoRALinear(getattr(parent, name), rank))


def export_adapter(model, path, metadata):
    values = {}
    for key in targets(model.config.num_hidden_layers, "ar"):
        module = model.get_submodule(key)
        values[key + ".lora_down.weight"] = module.A.detach().cpu().contiguous()
        values[key + ".lora_up.weight"] = module.B.detach().cpu().contiguous()
    temporary = str(path) + ".tmp"
    save_file(values, temporary, metadata={"format": "fl-yue2-lora-v1", "branch": "ar", **{k: str(v) for k, v in metadata.items()}})
    os.replace(temporary, path)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cursor_targets(item, tokenizer, prefix, device):
    words = np.load(item["cursor"], allow_pickle=False)
    request = SongRequest(item["style"], item["lyrics"], cot="off")
    text = request.text()
    begin_chars = text.index("[Lyrics]\n") + len("[Lyrics]\n")
    before = tokenizer.encode(text[:begin_chars])
    full = tokenizer.encode(text)
    if full[:len(before)] != before:
        raise ValueError("Lyric token boundary changed; rebuild alignment")
    lyric_ids = full[len(before):]
    offsets = [len(tokenizer.decode(lyric_ids[:k + 1])) for k in range(len(lyric_ids))]
    starts = [0] + offsets[:-1]
    word_tokens = [[k for k in range(len(lyric_ids)) if starts[k] < end and offsets[k] > start] for _, _, _, start, end in words]
    if not all(word_tokens):
        raise ValueError("Alignment does not match the exact lyrics")
    frames = len(item["codec"])
    word_index = np.clip(np.searchsorted(words[:, 0], np.arange(frames) / 25, side="right") - 1, 0, len(words) - 1)
    target = torch.zeros(frames, len(lyric_ids), device=device)
    for i, word in enumerate(word_index):
        target[i, word_tokens[word]] = 1 / len(word_tokens[word])
    return 1 + len(before), 1 + len(full), target


def clear_saved_outputs(run, export):
    for path in export.glob("step-*.safetensors"):
        path.unlink()
    for path in run.glob("preview-*.flac"):
        path.unlink()
    for name in ("resume.pt", "resume.tmp"):
        (run / name).unlink(missing_ok=True)


def train(request, emit, cancelled):
    torch.use_deterministic_algorithms(True)
    cfg = request["config"]
    assets = {key: value for key, value in request["assets"].items() if key != "download_missing"}
    mode, seed = cfg["mode"], cfg["seed"]
    if mode != "ar":
        raise ValueError("Only AR LoRA training is supported")
    prepared = read_json(request["dataset"])
    check_dataset(prepared)
    if prepared["head_hash"] != digest(Path(assets["head"])):
        raise ValueError("Tokenizer head changed; prepare the dataset again")
    generated = regularizer(assets["regularizer"])
    minted = [s for s in generated if s["src"] == "minted"]
    minted_val = [s for s in generated if s["src"] == "minted_val"][:6]
    songs = [{**s, "codec": np.load(s["tokens"], allow_pickle=False)} for s in prepared["songs"]]
    real = [s for s in songs if s["split"] == "train"]
    held = [s for s in songs if s["split"] == "validation"][:6]
    if not real or not minted or not held or not minted_val:
        raise ValueError("Training requires real and generated train/validation songs; reserve at least one song in each validation split")
    if cfg["cursor_weight"]:
        missing = [s["name"] for s in songs if s["lyrics"] and not s.get("cursor")]
        if missing:
            raise ValueError("Missing lyric alignment: " + ", ".join(missing))
    run = Path(request["run_directory"]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    export = Path(request["adapter_directory"]).resolve()
    export.mkdir(parents=True, exist_ok=True)
    asset_hashes = {key: digest(Path(assets[key])) for key in ("head", "regularizer", "initial_nar") if assets.get(key)}
    signature = fingerprint([prepared["fingerprint"], cfg, assets, asset_hashes])
    resume = request.get("resume", "")
    run_path = run / "run.json"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_model(assets["model"])
    device = next(model.parameters()).device
    install_lora(model, cfg["rank"])
    tokenizer = YuE2TextTokenizer(Path(assets["model"]) / "qwen.tiktoken")
    for item in songs + generated:
        item["prefix"] = token_prefixes(SongRequest(item["style"], item["lyrics"], cot="off"), tokenizer)
    cursor = nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False, device=device) if cfg["cursor_weight"] else None
    if cursor is not None:
        nn.init.eye_(cursor.weight)
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    groups = [{"params": [p for n, p in named.items() if n.endswith((".A", ".B"))], "lr": cfg["learning_rate"]}]
    if cursor is not None:
        groups.append({"params": list(cursor.parameters()), "lr": cfg["learning_rate"]})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0)
    base_lrs = [g["lr"] for g in groups]
    parameters = [p for g in groups for p in g["params"]]
    start_step = 0
    record = {"version": 1, "signature": signature, "mode": mode, "config": cfg, "assets": assets, "dataset": request["dataset"],
              "status": "running", "checkpoints": [], "metrics": [], "step": 0}
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Resume config, dataset, or model assets changed")
        for name, parameter in named.items():
            parameter.data.copy_(saved["model"][name].to(device))
        if cursor is not None:
            cursor.load_state_dict(saved["cursor"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start_step = saved["step"]
        record = read_run(run_path)
        record["metrics"] = [m for m in record["metrics"] if m["step"] <= start_step]
        record["checkpoints"] = [c for c in record["checkpoints"] if c["step"] <= start_step]
        record["status"] = "running"
    else:
        clear_saved_outputs(run, export)
    write_run(run_path, record)
    emit({"type": "status", "message": f"Training from step {start_step} of {cfg['steps']}"})
    elapsed = time.monotonic()

    def loss_for(item, training=True):
        prefix = item["prefix"]
        room = cfg["sequence_tokens"] - len(prefix) - 1
        if room < 1:
            raise ValueError("Caption/lyrics exceed training sequence budget")
        codec = item["codec"]
        if len(codec) > room and not cfg["allow_truncation"]:
            raise ValueError(f"{item['name']}: song exceeds sequence budget; increase it or explicitly allow truncation")
        body = [int(t) + CODEC_OFFSET for t in codec[:room]] + ([MUSIC_END] if len(codec) <= room else [])
        ids = torch.tensor([prefix + body], device=device)
        cur = cursor_targets(item, tokenizer, prefix, device) if cursor is not None and item.get("cursor") else None
        loss, auxiliary = ar_loss(model, ids, len(prefix), cur, cursor, checkpoint_layers=training)
        return loss + (cfg["cursor_weight"] * auxiliary if auxiliary is not None else 0), {"ce": float(loss.detach()), "cursor": float(auxiliary.detach()) if auxiliary is not None else 0.0}

    def save(step):
        if step > 0:
            path = export / f"step-{step:06d}.safetensors"
            acoustic = Path(assets["initial_nar"]).relative_to(export.parent).as_posix() if assets.get("initial_nar") else ""
            export_adapter(model, path, {"rank": cfg["rank"], "step": step, "base_revision": assets["model_revision"], "head_hash": prepared["head_hash"], "acoustic_adapter": acoustic})
            checkpoint_info = {"step": step, "adapter": str(path), "branch": "ar"}
        saved = {"signature": signature, "step": step, "model": {n: p.detach().cpu() for n, p in named.items()},
                 "cursor": cursor.state_dict() if cursor is not None else None,
                 "optimizer": optimizer.state_dict(), "rng": rng_state()}
        temporary = run / "resume.tmp"
        torch.save(saved, temporary)
        os.replace(temporary, run / "resume.pt")
        if step > 0:
            record["checkpoints"].append(checkpoint_info)
        write_run(run_path, record)
        if step > 0:
            emit({"type": "checkpoint", **checkpoint_info, "run": str(run_path)})

    try:
        if not resume and request.get("pause_at_checkpoint"):
            record["status"] = "preview"
            save(0)
            return str(run_path)
        for step in range(start_step + 1, cfg["steps"] + 1):
            cancelled()
            mult = min(1.0, step / max(1, cfg["warmup_steps"])) * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(step, cfg["schedule_steps"]) / cfg["schedule_steps"])))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * mult
            optimizer.zero_grad(set_to_none=True)
            report = {"loss": 0.0}
            for _ in range(cfg["accumulation"]):
                cancelled()
                item = random.choice(minted) if random.random() < cfg["generated_fraction"] else random.choice(real)
                loss, metrics = loss_for(item)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / cfg["accumulation"]).backward()
                report["loss"] += float(loss.detach()) / cfg["accumulation"]
                report.update(metrics)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            report.update({"type": "progress", "step": step, "max_steps": cfg["steps"], "lr": optimizer.param_groups[0]["lr"],
                           "seconds": time.monotonic() - elapsed, "gradient_norm": float(norm), "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
            record["step"] = step
            record["metrics"].append(report)
            emit(report)
            if step % cfg["save_every"] == 0 or step == cfg["steps"]:
                rng = rng_state()
                try:
                    random.seed(123)
                    np.random.seed(123)
                    torch.manual_seed(123)
                    for label, examples in (("artist_validation", held), ("generated_validation", minted_val)):
                        total = 0.0
                        for example in examples:
                            cancelled()
                            with torch.set_grad_enabled(False):
                                value, _ = loss_for(example, training=False)
                            total += float(value.detach())
                            del value
                        report[label] = total / len(examples)
                finally:
                    restore_rng(rng)
                save(step)
                if request.get("pause_at_checkpoint") and step < cfg["steps"]:
                    record["status"] = "preview"
                    write_run(run_path, record)
                    return str(run_path)
            if request.get("stop_after") == step:
                if step % cfg["save_every"]:
                    save(step)
                record["status"] = "paused"
                write_run(run_path, record)
                return str(run_path)
        record["status"] = "complete"
        write_run(run_path, record)
        return str(run_path)
    except BaseException as error:
        record["status"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
        record["error"] = str(error)
        write_run(run_path, record)
        raise
