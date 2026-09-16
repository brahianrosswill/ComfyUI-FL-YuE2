import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location("fl_yue2", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from fl_yue2.yue2.downloads import MODELS  # noqa: E402
from fl_yue2.yue2.training import downloads  # noqa: E402
from fl_yue2.yue2.training.prepare import prepare  # noqa: E402
from fl_yue2.yue2.training.trainer import train, load_model, install_lora  # noqa: E402
from fl_yue2.yue2.training.data import read_json  # noqa: E402
from fl_yue2.yue2.training.math import ar_loss  # noqa: E402
from fl_yue2.yue2.training.acoustic import acoustic_loss, fold_acoustic, prepare_targets  # noqa: E402
from fl_yue2.yue2.protocol import SongRequest, token_prefixes, CODEC_OFFSET, MUSIC_END  # noqa: E402
from fl_yue2.yue2.tokenizer import YuE2TextTokenizer  # noqa: E402


def event(value):
    if value["type"] != "progress" or value["step"] == value["max_steps"] or value["step"] % 5 == 0:
        print(json.dumps(value), flush=True)


def evaluate(assets, dataset, candidates, target_directory):
    prepared = read_json(dataset)
    songs = [{**item, "codec": np.load(item["tokens"], allow_pickle=False)} for item in prepared["songs"]]
    prepare_targets(songs, target_directory, event, lambda: None)
    held = [item for item in songs if item["split"] == "validation"]
    tokenizer = YuE2TextTokenizer(Path(assets["model"]) / "qwen.tiktoken")
    for item in held:
        item["off_prefix"] = token_prefixes(SongRequest(item["style"], item["lyrics"], cot="off"), tokenizer)
        item["score_prefix"] = token_prefixes(SongRequest(item["style"], item["lyrics"], cot="full"), tokenizer, item["abc_ids"])
    scores = {}
    for label, (ar_path, nar_path) in candidates.items():
        model = load_model(assets["model"])
        device = next(model.parameters()).device
        if ar_path:
            with safe_open(ar_path, framework="pt", device="cpu") as file:
                rank = file.get_tensor(next(key for key in file.keys() if key.endswith("lora_down.weight"))).shape[0]
            install_lora(model, rank)
            state = load_file(ar_path)
            for name in [key for key in state if key.endswith("lora_down.weight")]:
                module = model.get_submodule(name.removesuffix(".lora_down.weight"))
                module.A.data.copy_(state[name].to(device))
                module.B.data.copy_(state[name.replace("lora_down", "lora_up")].to(device))
        if nar_path:
            fold_acoustic(model, nar_path)
        values = {"direct_ar_ce": 0.0, "score_ar_ce": 0.0, "nar_flow": 0.0}
        with torch.set_grad_enabled(False):
            for index, item in enumerate(held):
                for key, prefix in (("direct_ar_ce", item["off_prefix"]), ("score_ar_ce", item["score_prefix"])):
                    body = [int(token) + CODEC_OFFSET for token in item["codec"]] + [MUSIC_END]
                    ids = torch.tensor([prefix + body], device=device)
                    loss, _ = ar_loss(model, ids, len(prefix), checkpoint_layers=False)
                    values[key] += float(loss)
                flow_item = {**item, "prefix": item["score_prefix"]}
                flows = [float(acoustic_loss(model, flow_item, 9000 + index, False, t=t, window_frames=1500, timestep="sigmoid")) for t in (.2, .5, .8)]
                values["nar_flow"] += sum(flows) / len(flows)
        scores[label] = {key: value / len(held) for key, value in values.items()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return scores


def run(steps):
    output = COMFY / "output" / "yue2_training"
    model = COMFY / "models" / "yue2" / "YuE2-3B"
    assets = {
        "model": str(model),
        "head": str(COMFY / "models" / "yue2" / "training_assets" / "tokenizer_head_joint_v4.pt"),
        "mert": str(COMFY / "models" / "yue2" / "MERT-v2-FullSong"),
        "regularizer": str(COMFY / "models" / "yue2" / "training_assets" / "minted_regularizer_pack.pt"),
        "initial_nar": str(COMFY / "models" / "loras" / "YuE2" / "pretrained" / "nar_lora_joint_v4.safetensors"),
        "model_revision": MODELS["YuE2-3B"],
        "mert_revision": MODELS["MERT-v2-FullSong"],
    }
    source = output / "smoke_sung" / "yue2_dataset.json"
    off = prepare({"dataset": str(source), "assets": assets, "score_planning": "off",
                   "cache_directory": str(output / "recipe_benchmark_off_features")}, event, lambda: None)
    sheet = downloads.sheetsage(True)
    full = prepare({"dataset": str(source), "assets": {**assets, "sheetsage": sheet}, "score_planning": "full",
                    "cache_directory": str(output / "recipe_benchmark_full_features")}, event, lambda: None)
    common = {"mode": "ar", "rank": 32, "learning_rate": 1e-4, "steps": steps, "save_every": steps,
              "sequence_tokens": 24576, "allow_truncation": False, "accumulation": 1, "seed": 42}
    requests = [
        ("legacy", off, {**common, "generated_fraction": .5, "cursor_weight": 0, "warmup_steps": 0,
                         "schedule_steps": steps, "train_acoustic": True}),
        ("joint", full, {**common, "recipe": "ai_toolkit_joint_v1", "weight_decay": 1e-4,
                         "ar_kl_weight": .2, "abc_dropout": .5, "ar_lr_multiplier": 1,
                         "acoustic_window_frames": 1500, "nar_start": "base"}),
    ]
    results = {}
    for name, dataset, config in requests:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_directory = output / f"recipe_benchmark_{name}"
        adapter_directory = COMFY / "models" / "loras" / "YuE2" / f"recipe_benchmark_{name}"
        print(f"START {name}", flush=True)
        path = train({"config": config, "assets": assets, "dataset": dataset,
                      "run_directory": str(run_directory), "adapter_directory": str(adapter_directory),
                      "pause_at_checkpoint": False}, event, lambda: None)
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        last = record["metrics"][-1]
        results[name] = {"run": path, "seconds": last["seconds"], "peak_gb": last["peak_gb"],
                         "metrics": {key: value for key, value in last.items() if key.endswith("validation")}}
    candidates = {"base": ("", ""),
                  "legacy": (str(COMFY / "models" / "loras" / "YuE2" / "recipe_benchmark_legacy" / f"step-{steps:06d}.safetensors"),
                             str(COMFY / "models" / "loras" / "YuE2" / "recipe_benchmark_legacy" / f"step-{steps:06d}-nar.safetensors")),
                  "joint": (str(COMFY / "models" / "loras" / "YuE2" / "recipe_benchmark_joint" / f"step-{steps:06d}.safetensors"),
                            str(COMFY / "models" / "loras" / "YuE2" / "recipe_benchmark_joint" / f"step-{steps:06d}-nar.safetensors"))}
    evaluation = evaluate(assets, full, candidates, output / "acoustic_targets")
    report = output / "recipe_benchmark.json"
    report.write_text(json.dumps({"steps": steps, "results": results, "evaluation": evaluation}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results, "evaluation": evaluation}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    run(parser.parse_args().steps)
