import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import comfy.lora
from comfy.model_patcher import ModelPatcher
from fl_yue2.runtime import MusicModel
from fl_yue2.adapters import patch_music
from torch import nn
from safetensors.torch import save_file, load_file

from fl_yue2.model import YuE2Model, StaticKVCache
from fl_yue2.training.math import hidden
from fl_yue2.training.trainer import install_lora, export_adapter, rng_state, restore_rng
from fl_yue2.training.data import dataset, read_json, write_json, fingerprint, run_name, regularizer, check_dataset
from fl_yue2.training.captioning import caption, validate_response


def tiny_model():
    return YuE2Model(dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, intermediate_size=48, vocab_size=64, rms_norm_eps=1e-6, rope_theta=10000,
                         latent_dim=64, max_latent_frames=100, timestep_shift=1.0), operations=nn)


def test_training_forward_matches_cached_inference_and_has_gradients():
    torch.manual_seed(4)
    model = tiny_model()
    inference = YuE2Model(vars(model.config))
    inference.load_state_dict(model.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    cache = StaticKVCache(2, 1, 2, 4, 8, torch.float32, "cpu")
    expected = inference(ids, cache, logits_to_keep=4).logits.detach()
    actual = model.lm_head(hidden(model, ids, checkpoint_layers=False))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.square().mean().backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_adapter_only_updates_and_export(tmp_path):
    model = tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model.model.layers[0].self_attn.q_proj.weight.clone()
    install_lora(model, 2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model.lm_head(hidden(model, torch.tensor([[1, 2, 3]]))).square().mean().backward()
    optimizer.step()
    projection = model.model.layers[0].self_attn.q_proj
    assert torch.equal(projection.base.weight, before)
    assert projection.B.abs().sum() > 0
    path = tmp_path / "adapter.safetensors"
    export_adapter(model, path, {})
    state = load_file(str(path))
    assert len(state) == 28
    assert all("lora_" in key for key in state)


def test_rng_resume_reproduces_draws():
    import random
    state = rng_state()
    first = random.random(), np.random.random(), torch.rand(3)
    restore_rng(state)
    second = random.random(), np.random.random(), torch.rand(3)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def songs(tmp_path):
    for index in range(3):
        sf.write(tmp_path / f"song{index}.wav", np.random.default_rng(index).normal(0, 0.1, 16000), 16000)
        (tmp_path / f"song{index}.caption.txt").write_text("piano", encoding="utf-8")
        (tmp_path / f"song{index}.lyrics.txt").write_text("", encoding="utf-8")


def test_dataset_requires_review_and_groups_song_split(tmp_path):
    songs(tmp_path)
    write_json(tmp_path / "song0.caption.json", {"reviewed": False})
    with pytest.raises(ValueError, match="review"):
        dataset(tmp_path, "", "", 0.2, 42)
    write_json(tmp_path / "song0.caption.json", {"reviewed": True, "reviewed_text": fingerprint(["piano", ""])})
    for index in (0, 1):
        (tmp_path / f"song{index}.song.txt").write_text("same song")
    result = read_json(dataset(tmp_path, "my_style", "", 0.2, 42))
    assert result["songs"][0]["split"] == result["songs"][1]["split"]
    assert {s["split"] for s in result["songs"]} == {"train", "validation"}
    assert all(s["style"] == "my_style, piano" for s in result["songs"])


@pytest.mark.parametrize("name", ["../escape", "x/y", "x\\y", "", "C:foo", "a.b"])
def test_run_containment(name):
    with pytest.raises(ValueError):
        run_name(name)


def test_token_only_regularizer_rejects_acoustic_mode(tmp_path):
    from fl_yue2.training.trainer import train
    with pytest.raises(ValueError, match="Only AR"):
        train({"config": {"mode": "nar", "seed": 0}, "assets": {}}, lambda _: None, lambda: None)


def test_caption_preserves_supplied_lyrics_and_cleans_upload(tmp_path):
    songs(tmp_path)
    for path in tmp_path.glob("*.caption.txt"):
        path.unlink()
    removed, uploaded = [], []
    def upload(file):
        uploaded.append(file)
        return SimpleNamespace(name="upload", state="ACTIVE", uri="https://example.invalid/audio", mime_type="audio/wav")
    result = dict(style="warm piano", lyrics="new guessed words", instrumental=False, uncertainty="", complete=True)
    client = SimpleNamespace(files=SimpleNamespace(upload=upload, delete=lambda name: removed.append(name)),
                             interactions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(result))))
    request = {"directory": str(tmp_path), "task": "both", "replace_existing": False, "model": "test", "output": str(tmp_path / "manifest.json")}
    caption(request, lambda _: None, lambda: None, client)
    assert len(uploaded) == len(removed) == 3
    assert (tmp_path / "song0.lyrics.txt").read_text() == ""
    assert (tmp_path / "song0.caption.txt").read_text() == "warm piano"
    caption(request, lambda _: None, lambda: None, client)
    assert len(uploaded) == 3


def test_caption_incomplete_is_not_accepted():
    with pytest.raises(ValueError, match="full recording"):
        validate_response(dict(style="piano", lyrics="", instrumental=True, uncertainty="", complete=False))


def test_dataset_rejects_changed_sidecars(tmp_path):
    songs(tmp_path)
    manifest = read_json(dataset(tmp_path, "", "", 0.2, 42))
    check_dataset(manifest)
    (tmp_path / "song0.lyrics.txt").write_text("changed words")
    with pytest.raises(ValueError, match="sidecars changed"):
        check_dataset(manifest)


def test_caption_segments_keep_repeated_lyrics_and_whole_song_style(tmp_path):
    sf.write(tmp_path / "song.wav", np.zeros(181 * 8000), 8000)
    responses = iter([
        dict(style="whole song rock", lyrics="partial", instrumental=False, uncertainty="", complete=False),
        dict(style="verse", lyrics="Sing it again", instrumental=False, uncertainty="", complete=True),
        dict(style="chorus", lyrics="Sing it again", instrumental=False, uncertainty="", complete=True),
    ])
    removed = []
    client = SimpleNamespace(files=SimpleNamespace(
        upload=lambda file: SimpleNamespace(name="upload", state="ACTIVE", uri="https://example.invalid/audio", mime_type="audio/flac"),
        delete=lambda name: removed.append(name)), interactions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(next(responses)))))
    caption({"directory": str(tmp_path), "task": "both", "replace_existing": False, "model": "test", "output": str(tmp_path / "manifest.json")}, lambda _: None, lambda: None, client)
    assert (tmp_path / "song.lyrics.txt").read_text() == "Sing it again\nSing it again"
    assert (tmp_path / "song.caption.txt").read_text() == "whole song rock"
    assert len(removed) == 3
    assert len(read_json(tmp_path / "song.caption.json")["segments"]) == 2


@pytest.mark.parametrize("always_incomplete", [False, True])
def test_caption_splits_incomplete_short_tracks(tmp_path, monkeypatch, always_incomplete):
    from fl_yue2.training import captioning
    sf.write(tmp_path / "song.wav", np.zeros(104 * 8000), 8000)
    durations = []
    def listen(client, audio, request, prompt, emit, cancelled):
        seconds = sf.info(audio).duration
        durations.append(seconds)
        return dict(style="dub", lyrics="Beat", instrumental=False, uncertainty="",
                    complete=not always_incomplete and seconds < 40)
    monkeypatch.setattr(captioning, "listen", listen)
    request = {"directory": str(tmp_path), "task": "both", "replace_existing": False,
               "model": "test", "output": str(tmp_path / "manifest.json")}
    if always_incomplete:
        with pytest.raises(ValueError, match="0.0-13.0s"):
            caption(request, lambda _: None, lambda: None, object())
        assert not (tmp_path / "song.lyrics.txt").exists()
        assert len(durations) == 4
    else:
        caption(request, lambda _: None, lambda: None, object())
        segments = read_json(tmp_path / "song.caption.json")["segments"]
        assert [(s["start"], s["end"]) for s in segments] == [(0, 26), (26, 52), (52, 78), (78, 104)]
        assert (tmp_path / "song.lyrics.txt").read_text() == "Beat\nBeat\nBeat\nBeat"
        assert durations[:3] == [104, 55, 29]


def test_adapter_strength_and_original_patcher_isolation(tmp_path):
    model = tiny_model()
    key = "model.layers.0.self_attn.q_proj"
    weight = model.get_submodule(key).weight.detach().clone()
    down, up = torch.randn(2, 32), torch.randn(32, 2)
    path = tmp_path / "partial.safetensors"
    save_file({key + ".lora_down.weight": down, key + ".lora_up.weight": up}, str(path), metadata={"format": "fl-yue2-lora-v1", "branch": "ar", "rank": "2"})
    original = MusicModel(ModelPatcher(model, torch.device("cpu"), torch.device("cpu")), None, None)
    for strength in (0.0, 1.0):
        changed = patch_music(original, [str(path)], ar_strength=strength)
        result = comfy.lora.calculate_weight(changed.patcher.patches[key + ".weight"], weight.clone(), key + ".weight")
        torch.testing.assert_close(result, weight + strength * (up @ down))
        assert not original.patcher.patches
        assert torch.equal(model.get_submodule(key).weight, weight)


@pytest.fixture
def saved_trainer_run(tmp_path, monkeypatch):
    from fl_yue2.training import nodes as training_nodes
    monkeypatch.setattr(training_nodes, "output_root", lambda: tmp_path)
    settings = {"style": "piano", "lyrics": "", "seed": 42, "max_seconds": 8}
    root = tmp_path / "saved"
    root.mkdir()
    (root / "one.flac").write_bytes(b"preview fixture")
    run = {"mode": "ar", "status": "complete", "assets": {"initial_nar": "paired-nar.safetensors"}, "checkpoints": [
        {"step": 1, "adapter": "ar-one.safetensors", "branch": "ar", "preview": "one.flac", "preview_settings": settings},
        {"step": 2, "adapter": "ar-two.safetensors", "branch": "ar", "preview": "one.flac", "preview_settings": settings}]}
    write_json(root / "run.json", run)
    inputs = dict(action="use_saved", output_name="saved", resume="", selected_step=0, render_previews=True,
                  preview_style="piano", preview_lyrics="", preview_seed=42, preview_seconds=8)
    return training_nodes, inputs


def test_saved_checkpoint_selection_does_not_run_workers(saved_trainer_run, monkeypatch):
    nodes, inputs = saved_trainer_run
    monkeypatch.setattr(nodes, "run_worker", lambda *args: pytest.fail("Saved selection ran a worker"))
    node = nodes.FL_YuE2_LoRATrainer()
    assert node.check_lazy_status("use_saved", assets=None, dataset=None, config=None) == []
    assert node.train(**inputs)["result"][0] == {"ar": "ar-two.safetensors", "nar": "paired-nar.safetensors"}
    inputs["selected_step"] = 1
    assert node.train(**inputs)["result"][0] == {"ar": "ar-one.safetensors", "nar": "paired-nar.safetensors"}


def test_changed_preview_only_runs_preview_worker(saved_trainer_run, monkeypatch):
    nodes, inputs = saved_trainer_run
    calls = []
    monkeypatch.setattr(nodes, "run_worker", lambda request, node_id: calls.append(request))
    inputs["preview_seed"] = 99
    nodes.FL_YuE2_LoRATrainer().train(**inputs)
    assert [c["operation"] for c in calls] == ["preview"]
    assert calls[0]["seed"] == 99


def test_training_outputs_selected_adapter(saved_trainer_run, monkeypatch):
    nodes, inputs = saved_trainer_run
    calls = []
    monkeypatch.setattr(nodes, "run_worker", lambda request, node_id: calls.append(request))
    inputs.update(action="train", assets={}, dataset="prepared.json", config={})
    node = nodes.FL_YuE2_LoRATrainer()
    assert node.check_lazy_status("train", assets=None, dataset=None, config=None) == ["assets", "dataset", "config"]
    result = node.train(**inputs)
    assert [c["operation"] for c in calls] == ["train"]
    assert result["ui"]["selected_step"] == [2]


def test_invalid_checkpoint_does_not_start_preview(saved_trainer_run, monkeypatch):
    nodes, inputs = saved_trainer_run
    monkeypatch.setattr(nodes, "run_worker", lambda *args: pytest.fail("Invalid selection ran a worker"))
    inputs["selected_step"] = 999
    with pytest.raises(ValueError, match="No checkpoint"):
        nodes.FL_YuE2_LoRATrainer().train(**inputs)


@pytest.mark.parametrize("previews", [True, False])
def test_training_interleaves_previews_and_resumes(saved_trainer_run, monkeypatch, previews):
    nodes, inputs = saved_trainer_run
    path = nodes.output_root() / "saved" / "run.json"
    calls = []

    def worker(request, node_id):
        calls.append(request)
        run = read_json(path)
        if request["operation"] == "train":
            run["status"] = "preview" if previews and not request["resume"] else "complete"
            if request["resume"]:
                assert request["resume"] == str(path.parent / "resume.pt")
                assert calls[-2]["operation"] == "preview"
        write_json(path, run)

    monkeypatch.setattr(nodes, "run_worker", worker)
    inputs.update(action="train", assets={}, dataset="prepared.json", config={}, render_previews=previews)
    nodes.FL_YuE2_LoRATrainer().train(**inputs)
    assert [c["operation"] for c in calls] == (["train", "preview", "train"] if previews else ["train"])
    assert calls[0]["pause_at_checkpoint"] is previews


@pytest.mark.parametrize("error", [RuntimeError, InterruptedError])
def test_failed_preview_does_not_resume_training(saved_trainer_run, monkeypatch, error):
    nodes, inputs = saved_trainer_run
    path = nodes.output_root() / "saved" / "run.json"
    calls = []

    def worker(request, node_id):
        calls.append(request["operation"])
        if request["operation"] == "preview":
            raise error("Preview failed")
        run = read_json(path)
        run["status"] = "preview"
        write_json(path, run)

    monkeypatch.setattr(nodes, "run_worker", worker)
    inputs.update(action="train", assets={}, dataset="prepared.json", config={})
    with pytest.raises(error, match="Preview failed"):
        nodes.FL_YuE2_LoRATrainer().train(**inputs)
    assert calls == ["train", "preview"]
    assert read_json(path)["checkpoints"]


def test_examples_have_direct_adapter_links():
    from pathlib import Path
    from fl_yue2.training.nodes import TRAINING_NODES
    assert "FL_YuE2_CheckpointCompare" not in TRAINING_NODES
    for path in (Path(__file__).parents[1] / "example_workflows").glob("*.json"):
        graph = read_json(path)
        nodes = {str(n["id"]): n for n in graph["nodes"]}
        assert all(n["type"] != "FL_YuE2_CheckpointCompare" for n in nodes.values())
        for link_id, origin, slot, target, target_slot, kind in graph["links"]:
            assert str(origin) in nodes and str(target) in nodes
            assert link_id in nodes[str(origin)]["outputs"][slot]["links"]
            assert nodes[str(target)]["inputs"][target_slot]["link"] == link_id
            if nodes[str(origin)]["type"] == "FL_YuE2_LoRATrainer":
                assert kind == "YUE2_ADAPTER"


def test_run_writer_waits_for_ui_reader(tmp_path, monkeypatch):
    from fl_yue2.training import data as training_data
    path = tmp_path / "run.json"
    training_data.write_run(path, {"step": 1})
    reading, release = Event(), Event()
    def held_read(target):
        with target.open(encoding="utf-8") as stream:
            reading.set()
            assert release.wait(3)
            return json.load(stream)
    monkeypatch.setattr(training_data, "read_json", held_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(training_data.read_run, path)
        assert reading.wait(2)
        writer = pool.submit(training_data.write_run, path, {"step": 2})
        try:
            with pytest.raises(FutureTimeout):
                writer.result(timeout=0.1)
        finally:
            release.set()
        assert reader.result(timeout=2) == {"step": 1}
        writer.result(timeout=2)
    assert training_data.read_run(path) == {"step": 2}


def test_missing_run_read_does_not_create_directories(tmp_path):
    from fl_yue2.training.data import read_run
    with pytest.raises(FileNotFoundError):
        read_run(tmp_path / "missing" / "run.json")
    assert not (tmp_path / "missing").exists()


def test_ar_regularizer_needs_tokens_only(tmp_path):
    np.save(tmp_path / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
    write_json(tmp_path / "regularizer.json", {"songs": [{"true_tokens": True, "tokens": str(tmp_path / "tokens.npy"), "split": "train"}]})
    assert regularizer(tmp_path / "regularizer.json")[0]["codec"].tolist() == [1, 2, 3]


def test_ar_loader_uses_paired_acoustic_metadata_and_rejects_nar(tmp_path, monkeypatch):
    from fl_yue2.training import nodes as training_nodes
    monkeypatch.setattr(training_nodes, "lora_root", lambda: tmp_path)
    save_file({"tensor": torch.zeros(1)}, str(tmp_path / "ar.safetensors"), metadata={"branch": "ar", "acoustic_adapter": "paired.safetensors"})
    save_file({"tensor": torch.zeros(1)}, str(tmp_path / "nar.safetensors"), metadata={"branch": "nar"})
    calls = []
    monkeypatch.setattr(training_nodes.adapters, "patch_music", lambda model, paths, strength: calls.append((paths, strength)))
    node = training_nodes.FL_YuE2_LoadLoRA()
    node.load(None, "ar.safetensors", 0.75)
    assert calls == [([str(tmp_path / "ar.safetensors"), str(tmp_path / "paired.safetensors")], 0.75)]
    assert node.INPUT_TYPES()["required"]["ar_adapter"][0] == ["none", "ar.safetensors"]
    with pytest.raises(ValueError, match="AR LoRA"):
        node.load(None, "nar.safetensors", 1.0)


def test_training_model_inputs_are_named_assets():
    from fl_yue2.training.nodes import FL_YuE2_TrainingModels
    schema = FL_YuE2_TrainingModels.INPUT_TYPES()
    assert set(schema["required"]) == {"tokenizer_head", "regularizer", "download_missing"}
    with pytest.raises(ValueError, match="Unknown"):
        FL_YuE2_TrainingModels().load("../head.pt", "minted_regularizer_pack.pt")


def test_overwrite_clears_only_saved_training_outputs(tmp_path):
    from fl_yue2.training.trainer import clear_saved_outputs
    run, export = tmp_path / "run", tmp_path / "adapters"
    run.mkdir()
    export.mkdir()
    for name in ("preview-000050-old.flac", "resume.pt", "resume.tmp", "notes.txt"):
        (run / name).write_bytes(b"old")
    for name in ("step-000050.safetensors", "step-000100.safetensors", "notes.txt"):
        (export / name).write_bytes(b"old")
    other = tmp_path / "other_run"
    other.mkdir()
    (other / "resume.pt").write_bytes(b"keep")
    clear_saved_outputs(run, export)
    assert sorted(p.name for p in run.iterdir()) == ["notes.txt"]
    assert sorted(p.name for p in export.iterdir()) == ["notes.txt"]
    assert (other / "resume.pt").read_bytes() == b"keep"


def test_train_always_runs_but_saved_selection_can_be_cached():
    from fl_yue2.training.nodes import FL_YuE2_LoRATrainer
    first = FL_YuE2_LoRATrainer.IS_CHANGED("train")
    second = FL_YuE2_LoRATrainer.IS_CHANGED("train")
    assert first != second
    assert FL_YuE2_LoRATrainer.IS_CHANGED("use_saved") == FL_YuE2_LoRATrainer.IS_CHANGED("use_saved")


def test_caption_requires_node_key_even_with_machine_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "machine-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "other-machine-key")
    with pytest.raises(ValueError, match="on the Gemini Music Captioner node"):
        caption({"api_key": "  "}, lambda _: None, lambda: None)


def test_caption_uses_node_key(monkeypatch, tmp_path):
    from google import genai
    monkeypatch.setenv("GEMINI_API_KEY", "machine-key")
    captured = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.append(kwargs) or SimpleNamespace(close=lambda: None))
    songs(tmp_path)
    caption({"api_key": "  node-key  ", "directory": str(tmp_path), "task": "both",
             "replace_existing": False, "output": str(tmp_path / "manifest.json")}, lambda _: None, lambda: None)
    assert captured == [{"api_key": "node-key"}] * 3
    assert "node-key" not in (tmp_path / "manifest.json").read_text()


def test_caption_worker_transmits_key_without_saving_it(monkeypatch, tmp_path):
    import io
    from fl_yue2.training import service
    writes, calls = [], []
    process = SimpleNamespace(
        stdin=SimpleNamespace(write=writes.append, close=lambda: None),
        stdout=io.StringIO('YUE2_EVENT {"type":"complete","result":"manifest.json"}\n'),
        returncode=0, wait=lambda **kwargs: 0, poll=lambda: 0,
    )
    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return process
    monkeypatch.setattr(service, "output_root", lambda: tmp_path)
    monkeypatch.setattr(service.subprocess, "Popen", launch)
    monkeypatch.setattr(service.mm, "soft_empty_cache", lambda: None)
    result = service.run_worker({"operation": "caption"}, api_key="node-secret")
    assert result == "manifest.json"
    assert json.loads(writes[0]) == "node-secret"
    assert calls[0][1]["stdin"] == service.subprocess.PIPE
    assert "node-secret" not in str(calls)
    for path in (tmp_path / "jobs").iterdir():
        assert "node-secret" not in path.read_text()


def test_caption_node_passes_key_separately(monkeypatch, tmp_path):
    from fl_yue2.training import nodes
    captured = []
    def run(request, node_id=None, **kwargs):
        captured.append((request, kwargs))
        return "manifest.json"
    monkeypatch.setattr(nodes, "run_worker", run)
    node = nodes.FL_YuE2_GeminiMusicCaptioner()
    node.caption(str(tmp_path), "test", "both", False, "", api_key=" node-key ")
    assert captured[0][1] == {"api_key": "node-key"}
    assert "api_key" not in captured[0][0]
    with pytest.raises(ValueError, match="Enter a Google API key"):
        node.caption(str(tmp_path), "test", "both", False, "", api_key="")
