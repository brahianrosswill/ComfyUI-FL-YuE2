import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from fl_yue2.yue2 import nar
from fl_yue2.yue2.adapters import targets
from fl_yue2.yue2.training import paired_data, paired_models, paired_prepare, acoustic, trainer
from fl_yue2.yue2.training.trainer import install_lora
from fl_yue2.yue2.training import nodes as training_nodes
from fl_yue2.yue2.training import math as training_math
from fl_yue2.yue2.training.data import read_json, write_json
from fl_yue2.yue2.training.paired_nodes import FL_YuE2_PairedDataset, FL_YuE2_AudioAdapterConfig
from fl_yue2.yue2.training.prepare import predict
from fl_yue2.yue2.model import YuE2Model
from fl_yue2.yue2.conditioning import AudioConditioner


def tiny_model():
    model = YuE2Model(dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, intermediate_size=48, vocab_size=64, rms_norm_eps=1e-6, rope_theta=10000,
                          latent_dim=64, max_latent_frames=100, max_position_embeddings=256, timestep_shift=1.0), operations=nn)
    for parameter in model.parameters():
        parameter.data.normal_(0, .1)
    return model


def pairs(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for i in range(4):
        audio = np.random.default_rng(i).normal(0, .1, 16000)
        sf.write(source / f"song{i}.wav", audio, 16000)
        sf.write(target / f"song{i}.wav", audio * .5, 16000)
        (target / f"song{i}.song.txt").write_text(str(i // 2))
    return source, target


def test_pairs_split_by_song_and_detect_changed_source(tmp_path):
    source, target = pairs(tmp_path)
    data = paired_data.make_dataset(source, target, "music", .25, 42)
    assert data["pairs"][0]["split"] == data["pairs"][1]["split"]
    assert {p["split"] for p in data["pairs"]} == {"train", "validation"}
    assert all(p["source_hash"] != p["target_hash"] for p in data["pairs"])
    paired_data.check_pairs(data)
    sf.write(source / "song0.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="source audio changed"):
        paired_data.check_pairs(data)


def test_pairs_reject_unaligned_targets(tmp_path):
    source, target = pairs(tmp_path)
    sf.write(target / "song0.wav", np.zeros(17000), 16000)
    with pytest.raises(ValueError, match="durations differ"):
        paired_data.make_dataset(source, target, "music", .25, 42)


def test_feature_extraction_preserves_short_last_chunk(monkeypatch):
    calls = []
    def features(model, processor, samples, cancelled):
        calls.append(len(samples))
        return np.zeros((len(samples) // 960, 1024), np.float32)
    monkeypatch.setattr(paired_prepare, "mert_features", features)
    output = paired_prepare.extract_features(None, None, np.zeros((31 * 48000 - 24000, 2), np.float32), lambda: None)
    assert len(output) == 762
    assert calls == [720000, 24000]


def test_training_seeds_do_not_randomize_after_queue():
    for node in (FL_YuE2_PairedDataset, FL_YuE2_AudioAdapterConfig):
        assert node.INPUT_TYPES()["required"]["seed"][1]["control_after_generate"] is False


def test_prediction_uses_normalized_features_and_full_head_window():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(1, 512, 1))

        def forward(self, value):
            assert value.shape == (1, 512, 2)
            assert torch.equal(value[0, :200], torch.tensor([2., -1.]).expand(200, 2))
            assert not value[0, 200:].any()
            return value

    result = predict(Head(), np.tile([2., -1.], (200, 1)), lambda: None, input_normalized=True)
    assert np.array_equal(result, np.zeros(200, dtype=np.int32))


def test_flow_matches_cached_inference_and_backpropagates_to_source(monkeypatch):
    monkeypatch.setattr(training_math, "MUSIC_END", 6)
    model = tiny_model().requires_grad_(False)
    prefix, ids = [1, 2], [3, 4, 5]
    source = model.model.embed_tokens(torch.tensor(ids)).detach().requires_grad_()
    target, noise = torch.randn(3, 64), torch.randn(3, 64)
    t = .4
    inference_model = YuE2Model(vars(model.config))
    inference_model.load_state_dict(model.state_dict())
    inference = nar.CachedNAR(inference_model, nar.Chunk(prefix + ids + [6], noise))
    expected = (inference.velocity(t * noise + (1 - t) * target, math.log(t / (1 - t))) - (noise - target)).square().mean()
    actual = training_math.flow_loss(model, prefix, source, target, t, noise, True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.backward()
    assert source.grad.isfinite().all() and source.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_export_sums_starting_and_trained_acoustic_adapters(tmp_path):
    model = tiny_model().requires_grad_(False)
    base = tmp_path / "base"
    base.mkdir()
    save_file(model.state_dict(), str(base / "model.safetensors"))
    initial = {}
    expected = {}
    for key in targets(model.config.num_hidden_layers, "nar"):
        module = model.get_submodule(key)
        a, b = torch.randn(2, module.in_features), torch.randn(module.out_features, 2)
        initial[key + ".lora_down.weight"], initial[key + ".lora_up.weight"] = a, b
        wrapped = training_math.LoRALinear(module, 3)
        wrapped.B.data.normal_()
        parent, name = key.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, wrapped)
        expected[key] = b @ a + wrapped.B.detach() @ wrapped.A.detach()
    model.vae2llm.weight.data.add_(.25)
    head = nn.Linear(2, 2)
    path = paired_models.export_bundle(tmp_path / "export", model, head, initial,
                                       {"model": "hash", "mert_revision": "m", "vae_revision": "v"},
                                       {"model": str(base)}, "acoustic", 3)
    manifest = read_json(path)
    state = load_file(str(tmp_path / "export" / manifest["acoustic"]))
    for key, delta in expected.items():
        torch.testing.assert_close(state[key + ".lora_up.weight"] @ state[key + ".lora_down.weight"], delta)
    torch.testing.assert_close(state["vae2llm.diff"], torch.full_like(model.vae2llm.weight, .25))
    assert set(manifest["files"]) == {"head.safetensors", "acoustic.safetensors"}


def test_conditioned_flow_matches_inference_and_trains_projections(monkeypatch):
    monkeypatch.setattr(training_math, "MUSIC_END", 6)
    model = tiny_model().requires_grad_(False)
    inference_model = YuE2Model(vars(model.config))
    inference_model.load_state_dict(model.state_dict())
    condition = AudioConditioner(32, (0, 1), nn)
    for parameter in condition.parameters():
        nn.init.zeros_(parameter)
    prefix, ids = [1, 2], [3, 4, 5]
    source = model.model.embed_tokens(torch.tensor(ids)).detach()
    target, noise, audio = torch.randn(3, 64), torch.randn(3, 64), torch.randn(3, 64)
    chunk = nar.Chunk(prefix + ids + [6], noise)
    plain = nar.CachedNAR(inference_model, chunk)
    zero = nar.CachedNAR(inference_model, chunk, condition(audio))
    torch.testing.assert_close(plain.velocity(noise, 0.), zero.velocity(noise, 0.), rtol=0, atol=0)
    for parameter in condition.parameters():
        nn.init.normal_(parameter, std=.1)
    projected = condition(audio)
    t = .4
    engine = nar.CachedNAR(inference_model, chunk, projected)
    expected = (engine.velocity(t * noise + (1-t) * target, math.log(t / (1-t))) - (noise-target)).square().mean()
    actual = training_math.flow_loss(model, prefix, source, target, t, noise, False, True, projected)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in condition.parameters())
    assert all(p.grad is None for p in model.parameters())
    unconditioned = condition(torch.zeros_like(audio))
    guided = nar.CachedNAR(inference_model, chunk, projected, unconditioned, 0.)
    reference = nar.CachedNAR(inference_model, chunk, unconditioned)
    torch.testing.assert_close(guided.velocity(noise, 0.), reference.velocity(noise, 0.))


def test_conditioned_bundle_requires_all_weights_and_checks_hashes(tmp_path, monkeypatch):
    model = tiny_model()
    base = tmp_path / "base"
    base.mkdir()
    save_file(model.state_dict(), str(base / "model.safetensors"))
    identity = {"model": paired_models.digest(base / "model.safetensors"),
                "mert_revision": paired_models.MODELS["MERT-v2-FullSong"], "vae_revision": paired_models.MODELS["YuE2-Vae"]}
    conditioner = AudioConditioner(32, (0, 7, 14, 21), nn)
    path = paired_models.export_bundle(tmp_path / "export", model, nn.Linear(2, 2), {}, identity, {}, "head", 0, conditioner)
    monkeypatch.setattr(paired_models, "resolve", lambda *args: base)
    manifest, _, _, _ = paired_models.load_bundle(path)
    restored = AudioConditioner(32, (0, 7, 14, 21), nn)
    restored.load_state_dict(load_file(str(tmp_path / "export" / "conditioning.safetensors")))
    audio = torch.randn(3, 64)
    for layer, value in conditioner(audio).items():
        torch.testing.assert_close(restored(audio)[layer], value)
    weight_path = tmp_path / "export" / "conditioning.safetensors"
    original = weight_path.read_bytes()
    weight_path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="changed or is incomplete"):
        paired_models.load_bundle(path)
    weight_path.write_bytes(original)
    manifest["mode"] = "conditioned"
    del manifest["conditioning"]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="missing its source projections"):
        paired_models.load_bundle(path)


def test_conditioning_chunks_follow_source_frame_offsets(monkeypatch):
    seen = []
    conditioner = AudioConditioner(32, (0,), nn)
    def capture(module, args):
        seen.append(args[0].clone())
    conditioner.register_forward_pre_hook(capture)
    monkeypatch.setattr(nar, "song_chunks", lambda *args: [nar.Chunk([1], torch.zeros(n, 64)) for n in (2, 3)])
    class Engine:
        def __init__(self, model, chunk, *args):
            self.chunk = chunk
        def solve(self, *args):
            return self.chunk.noise
        def close(self):
            pass
    monkeypatch.setattr(nar, "CachedNAR", Engine)
    source = torch.randn(5, 64)
    result = nar.synthesize(None, [1], [0] * 5, 42, conditioner=conditioner, condition=source)
    assert result.shape == (5, 64)
    torch.testing.assert_close(torch.cat(seen), source)


def test_acoustic_objective_only_updates_decoder_and_preserves_rng(monkeypatch):
    monkeypatch.setattr(training_math, "MUSIC_END", 6)
    monkeypatch.setattr(acoustic, "CODEC_OFFSET", 0)
    model = tiny_model().requires_grad_(False)
    install_lora(model, 2)
    ar_parameters = [p for p in model.parameters() if p.requires_grad]
    acoustic.install_acoustic(model, 2)
    item = {"codec": np.array([3, 4, 5], dtype=np.int32), "latents": np.random.default_rng(4).normal(size=(3, 64)).astype(np.float32), "prefix": [1, 2]}
    before = torch.get_rng_state().clone()
    loss = acoustic.acoustic_loss(model, item, 42)
    loss.backward()
    assert torch.equal(torch.get_rng_state(), before)
    assert all(p.grad is None for p in ar_parameters)
    assert model.llm2vae.weight.grad.abs().sum() > 0
    assert model.model.layers[0].nar_self_attn.q_proj.B.grad.abs().sum() > 0
    torch.testing.assert_close(acoustic.acoustic_loss(model, item, 42), loss)


def test_original_trainer_acoustic_resume_and_ar_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(training_math, "MUSIC_END", 6)
    monkeypatch.setattr(acoustic, "CODEC_OFFSET", 0)
    monkeypatch.setattr(trainer, "CODEC_OFFSET", 0)
    monkeypatch.setattr(trainer, "MUSIC_END", 6)
    base = tiny_model().requires_grad_(False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(base.state_dict(), str(model_dir / "model.safetensors"))
    def load_model(path):
        result = tiny_model().requires_grad_(False)
        result.load_state_dict(base.state_dict())
        return result
    monkeypatch.setattr(trainer, "load_model", load_model)
    monkeypatch.setattr(trainer, "YuE2TextTokenizer", lambda *args: None)
    monkeypatch.setattr(trainer, "token_prefixes", lambda *args: [1, 2])
    monkeypatch.setattr(trainer, "check_dataset", lambda *args: None)
    codec = np.array([3, 4, 5], dtype=np.int32)
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, codec)
    head = tmp_path / "head"
    head.write_bytes(b"head")
    regularizer = tmp_path / "regularizer"
    regularizer.write_bytes(b"regularizer")
    songs = [{"name": split, "tokens": str(tokens), "style": "music", "lyrics": "", "split": split} for split in ("train", "validation")]
    dataset = tmp_path / "dataset.json"
    write_json(dataset, {"fingerprint": "dataset", "head_hash": trainer.digest(head), "songs": songs})
    monkeypatch.setattr(trainer, "regularizer", lambda _: [{"name": src, "style": "music", "lyrics": "", "src": src, "codec": codec} for src in ("minted", "minted_val")])
    def targets(songs, *args):
        for item in songs:
            item["latents"] = np.random.default_rng(5).normal(size=(3, 64)).astype(np.float32)
    monkeypatch.setattr(trainer, "prepare_targets", targets)
    assets = {"model": str(model_dir), "head": str(head), "regularizer": str(regularizer), "model_revision": "test"}
    config = {"mode": "ar", "seed": 42, "rank": 2, "cursor_weight": 0, "learning_rate": .001, "steps": 4,
              "save_every": 2, "sequence_tokens": 32, "allow_truncation": False, "warmup_steps": 0,
              "schedule_steps": 4, "accumulation": 1, "generated_fraction": .5}
    def request(name, enabled):
        return {"config": {**config, "train_acoustic": enabled}, "assets": assets, "dataset": str(dataset),
                "run_directory": str(tmp_path / name), "adapter_directory": str(tmp_path / "exports" / name), "pause_at_checkpoint": False}
    for name, enabled in (("ar", False), ("joint", True)):
        trainer.train(request(name, enabled), lambda _: None, lambda: None)
    ar = load_file(str(tmp_path / "exports/ar/step-000004.safetensors"))
    joint = load_file(str(tmp_path / "exports/joint/step-000004.safetensors"))
    assert all(torch.equal(ar[key], joint[key]) for key in ar)
    resumed = request("resumed", True)
    trainer.train({**resumed, "stop_after": 2}, lambda _: None, lambda: None)
    trainer.train({**resumed, "resume": str(tmp_path / "resumed/resume.pt")}, lambda _: None, lambda: None)
    for filename in ("step-000004.safetensors", "step-000004-nar.safetensors"):
        first = load_file(str(tmp_path / "exports/joint" / filename))
        second = load_file(str(tmp_path / "exports/resumed" / filename))
        assert all(torch.equal(first[key], second[key]) for key in first)
    run = read_json(tmp_path / "joint/run.json")
    assert run["metrics"][-1]["acoustic_validation"] > 0
    assert Path(run["checkpoints"][-1]["acoustic_adapter"]).is_file()
    monkeypatch.setattr(training_nodes, "lora_root", lambda: tmp_path / "exports")
    captured = []
    monkeypatch.setattr(training_nodes.adapters, "patch_music", lambda model, paths, strength: captured.append(paths))
    training_nodes.FL_YuE2_LoadLoRA().load(None, "joint/step-000004.safetensors", 1.0)
    assert [Path(path) for path in captured[0]] == [tmp_path / "exports/joint/step-000004.safetensors", tmp_path / "exports/joint/step-000004-nar.safetensors"]
