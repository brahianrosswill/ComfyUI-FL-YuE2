"""Source-conditioned acoustic flow training with resumable head/NAR adapters."""
import os
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..protocol import SongRequest, token_prefixes, CODEC_OFFSET, CODEC_SIZE
from ..tokenizer import YuE2TextTokenizer
from ..conditioning import AudioConditioner
from .data import read_json, read_run, write_run, fingerprint
from .math import flow_loss
from .paired_data import check_pairs, asset_identity
from .paired_models import initialize, export_bundle
from .prepare import normalize
from .trainer import rng_state, restore_rng


def train_pairs(request, emit, cancelled):
    cfg, assets = request["config"], request["assets"]
    if cfg["mode"] not in ("head", "acoustic", "joint", "conditioned"):
        raise ValueError("Choose head, acoustic, joint, or conditioned audio-adapter training")
    data = read_json(request["dataset"])
    check_pairs(data)
    identity = asset_identity(assets)
    if data.get("identity") != identity:
        raise ValueError("Audio training assets changed; prepare the pairs again")
    signature = fingerprint([data["fingerprint"], cfg, identity])
    root, export = Path(request["run_directory"]), Path(request["adapter_directory"])
    root.mkdir(parents=True, exist_ok=True)
    export.mkdir(parents=True, exist_ok=True)
    path = root / "run.json"
    resume = request.get("resume", "")
    if not resume and path.exists():
        raise ValueError("Audio adapter run already exists. Resume it or choose a new output name.")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model, head, initial = initialize(assets, cfg["mode"], cfg["rank"])
    conditioner = None
    if cfg["mode"] == "conditioned":
        conditioner = AudioConditioner(model.config.hidden_size, (0, 7, 14, 21), torch.nn).cuda()
        for parameter in conditioner.parameters():
            torch.nn.init.zeros_(parameter)
    tokenizer = YuE2TextTokenizer(Path(assets["model"]) / "qwen.tiktoken")
    head_training = cfg["mode"] in ("head", "joint")
    named = {"head." + k: v for k, v in head.named_parameters() if v.requires_grad}
    named.update({"model." + k: v for k, v in model.named_parameters() if v.requires_grad})
    if conditioner is not None:
        named.update({"conditioner." + k: v for k, v in conditioner.named_parameters()})
    groups = []
    if conditioner is not None:
        io_names = {"model." + module + "." + key for module in ("vae2llm", "llm2vae") for key in ("weight", "bias")}
        groups = [{"params": [v for k, v in named.items() if k.startswith("model.") and k not in io_names], "lr": cfg["acoustic_learning_rate"]},
                  {"params": [v for k, v in named.items() if k in io_names], "lr": cfg.get("projection_learning_rate", .00002)},
                  {"params": list(conditioner.parameters()), "lr": cfg.get("condition_learning_rate", .001)}]
    else:
        for prefix, lr in (("head.", cfg["head_learning_rate"]), ("model.", cfg["acoustic_learning_rate"])):
            parameters = [v for k, v in named.items() if k.startswith(prefix)]
            if parameters:
                groups.append({"params": parameters, "lr": lr})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=0)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    rows = []
    for pair in data["pairs"]:
        row = {**pair, **{k: np.load(pair[k], allow_pickle=False) for k in ("features", "latents", "source_latents", "anchor_tokens")}}
        row["features"] = normalize(row["features"])
        row["prefix"] = token_prefixes(SongRequest(pair["style"], pair["lyrics"], cot="off"), tokenizer)
        if len(row["features"]) < cfg["window_frames"]:
            raise ValueError(f"{pair['name']}: shorter than the training window")
        rows.append(row)
    train = [p for p in rows if p["split"] == "train"]
    held = [p for p in rows if p["split"] == "validation"]
    if not train or not held:
        raise ValueError("Paired training requires separate train and validation songs")
    record = {"version": 1, "mode": "audio", "signature": signature, "config": cfg, "assets": assets,
              "dataset": request["dataset"], "status": "running", "step": 0, "metrics": [], "checkpoints": []}
    start = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Resume dataset, configuration, or base assets changed")
        with torch.set_grad_enabled(False):
            for name, parameter in named.items():
                parameter.copy_(saved["parameters"][name].to(parameter))
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start = saved["step"]
        record = read_run(path)
        record["metrics"] = [r for r in record["metrics"] if r["step"] <= start]
        record["checkpoints"] = [c for c in record["checkpoints"] if c["step"] <= start]
        record["status"] = "complete" if start >= cfg["steps"] else "running"
    write_run(path, record)
    embedding = model.model.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE]
    frames = cfg["window_frames"]

    def loss_for(row, start_frame, training, replay=False, source=None, evaluation_t=.5, zero_condition=False):
        cancelled()
        source = row if source is None else source
        token_source = row if conditioner is not None else source
        features = torch.from_numpy(token_source["features"][start_frame:start_frame + frames].copy())[None].cuda()
        target = torch.from_numpy(row["source_latents" if replay else "latents"][start_frame:start_frame + frames].copy()).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.set_grad_enabled(training and head_training):
                logits = head(F.pad(features, (0, 0, 0, head.pos.shape[1] - frames)))[0, :frames].float()
                ids = logits.argmax(-1)
                if training and head_training:
                    probabilities = logits.softmax(-1)
                    hard = F.one_hot(ids, CODEC_SIZE).float()
                    encoded = (hard + probabilities - probabilities.detach()).to(embedding.dtype) @ embedding
                else:
                    encoded = embedding[ids]
            generator = None if training else torch.Generator(device="cuda").manual_seed(1234)
            noise = torch.randn(target.shape, device="cuda", generator=generator)
            t = float(np.clip(np.random.beta(2, 2), .02 if conditioner is not None else .05, .98 if conditioner is not None else .95)) if training else evaluation_t
            projected = None
            if conditioner is not None:
                condition = torch.from_numpy(source["source_latents"][start_frame:start_frame + frames].copy()).cuda()
                if zero_condition or (training and random.random() < cfg.get("condition_dropout", .1)):
                    condition = torch.zeros_like(condition)
                projected = conditioner(condition)
            flow = flow_loss(model, row["prefix"], encoded, target, t, noise, head_training, training, projected)
            anchor = flow.new_zeros(())
            if training and head_training and cfg["anchor_weight"]:
                labels = torch.from_numpy(row["anchor_tokens"][start_frame:start_frame + frames].astype(np.int64)).cuda()
                anchor = F.cross_entropy(logits, labels)
        return flow + cfg["anchor_weight"] * anchor

    def evaluate():
        head.eval()
        with torch.set_grad_enabled(False):
            matched, wrong, zero = [], [], []
            for index, row in enumerate(held[:4]):
                other = held[(index + 1) % len(held)] if len(held) > 1 else train[0]
                for t in ((.2, .5, .8) if conditioner is not None else (.5,)):
                    matched.append(float(loss_for(row, 0, False, evaluation_t=t)))
                    wrong.append(float(loss_for(row, 0, False, source=other, evaluation_t=t)))
                    if conditioner is not None:
                        zero.append(float(loss_for(row, 0, False, evaluation_t=t, zero_condition=True)))
        head.train(head_training)
        metrics = {"paired_flow": sum(matched) / len(matched), "wrong_source_flow": sum(wrong) / len(wrong)}
        if zero:
            metrics["no_condition_flow"] = sum(zero) / len(zero)
        return metrics

    def save(step):
        bundle = export_bundle(export / f"step-{step:06d}", model, head, initial, identity, assets, cfg["mode"], step, conditioner)
        saved = {"signature": signature, "step": step, "parameters": {k: v.detach().cpu() for k, v in named.items()},
                 "optimizer": optimizer.state_dict(), "rng": rng_state()}
        temporary = root / "resume.tmp"
        torch.save(saved, temporary)
        os.replace(temporary, root / "resume.pt")
        item = {"step": step, "bundle": bundle}
        if step:
            record["checkpoints"].append(item)
        else:
            record["baseline"] = item
        write_run(path, record)
        emit({"type": "checkpoint", "step": step, "run": str(path)})

    elapsed = time.monotonic()
    try:
        if not resume:
            record["baseline_metrics"] = evaluate()
            record["status"] = "preview" if request["pause_at_checkpoint"] else "running"
            save(0)
            if request["pause_at_checkpoint"]:
                return str(path)
        head.train(head_training)
        for step in range(start + 1, cfg["steps"] + 1):
            cancelled()
            for group, lr in zip(optimizer.param_groups, base_lrs):
                decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / cfg["steps"])) if conditioner is not None else 1
                group["lr"] = lr * min(1, step / max(1, cfg["warmup_steps"])) * decay
            optimizer.zero_grad(set_to_none=True)
            total = 0
            for _ in range(cfg["accumulation"]):
                row = random.choice(train)
                offset = random.randint(0, len(row["features"]) - frames)
                loss = loss_for(row, offset, True, random.random() < cfg["source_replay"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite paired training loss")
                (loss / cfg["accumulation"]).backward()
                total += float(loss.detach()) / cfg["accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1, error_if_nonfinite=True)
            optimizer.step()
            row = {"type": "progress", "step": step, "max_steps": cfg["steps"], "loss": total,
                   "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm), "seconds": time.monotonic() - elapsed,
                   "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            record["step"] = step
            record["metrics"].append(row)
            if step % cfg["save_every"] == 0 or step == cfg["steps"]:
                state = rng_state()
                try:
                    validation = evaluate()
                    row.update({"artist_validation": validation.pop("paired_flow"), **validation})
                    if row["artist_validation"] < record.get("best_validation", float("inf")):
                        record.update({"best_step": step, "best_validation": row["artist_validation"]})
                finally:
                    restore_rng(state)
                record["status"] = "complete" if step == cfg["steps"] else "preview" if request["pause_at_checkpoint"] else "running"
                save(step)
            emit(row)
            if record["status"] in ("preview", "complete"):
                return str(path)
        return str(path)
    except BaseException as error:
        record["status"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
        record["error"] = str(error)
        write_run(path, record)
        raise
