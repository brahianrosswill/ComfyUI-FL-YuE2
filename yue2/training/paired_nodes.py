from pathlib import Path
import uuid

import folder_paths
import soundfile as sf
import torch

from .data import contained, run_name, write_json, read_run
from .paired_data import make_dataset
from .service import output_root, run_worker


def adapter_root():
    return Path(folder_paths.get_folder_paths("loras")[0]) / "YuE2_audio"


class FL_YuE2_PairedDataset:
    CATEGORY = "FL YuE2/Audio Training"
    FUNCTION = "make"
    RETURN_TYPES = ("YUE2_AUDIO_PAIRS",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_directory": ("STRING", {"default": "yue2_pairs/source", "tooltip": "Aligned input audio. Relative to ComfyUI input, or an explicit folder."}),
            "target_directory": ("STRING", {"default": "yue2_pairs/target", "tooltip": "Desired output audio with matching filename stems and durations. Target .caption.txt, .lyrics.txt and .song.txt sidecars are optional."}),
            "default_style": ("STRING", {"default": "instrumental electronic music", "multiline": True, "tooltip": "Target style when a pair has no caption sidecar."}),
            "validation_fraction": ("FLOAT", {"default": .2, "min": .01, "max": .5, "step": .01, "tooltip": "Song groups reserved for held-out validation."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "control_after_generate": False, "tooltip": "Stable song-group split seed. Keep it unchanged when resuming."})}}

    def make(self, source_directory, target_directory, default_style, validation_fraction, seed):
        source = Path(folder_paths.get_input_directory()) / source_directory
        target = Path(folder_paths.get_input_directory()) / target_directory
        data = make_dataset(source, target, default_style, validation_fraction, seed)
        path = output_root() / "paired_datasets" / (data["fingerprint"][:24] + ".json")
        write_json(path, data)
        return (str(path),)


class FL_YuE2_PrepareAudioPairs:
    CATEGORY = "FL YuE2/Audio Training"
    FUNCTION = "prepare"
    RETURN_TYPES = ("YUE2_PREPARED_AUDIO_PAIRS",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dataset": ("YUE2_AUDIO_PAIRS",), "assets": ("YUE2_TRAINING_ASSETS",),
                             "cache_directory": ("STRING", {"default": "paired_features", "tooltip": "Prepared source features and target latents under output/yue2_training."})},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    def prepare(self, dataset, assets, cache_directory, unique_id=None):
        return (run_worker({"operation": "paired_prepare", "dataset": dataset, "assets": assets,
                            "cache_directory": str(contained(output_root(), cache_directory))}, unique_id),)


class FL_YuE2_AudioAdapterConfig:
    CATEGORY = "FL YuE2/Audio Training"
    FUNCTION = "configure"
    RETURN_TYPES = ("YUE2_AUDIO_TRAIN_CONFIG",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["acoustic", "head", "joint", "conditioned"], {"tooltip": "Conditioned: acoustic LoRA plus continuous source audio at four decoder layers. Other modes train the acoustic adapter, source head, or both."}),
            "steps": ("INT", {"default": 1000, "min": 1, "max": 100000, "tooltip": "Total optimizer updates, including resumed steps."}),
            "save_every": ("INT", {"default": 100, "min": 1, "max": 100000, "tooltip": "Checkpoint and held-out evaluation interval."}),
            "rank": ("INT", {"default": 16, "min": 1, "max": 128, "tooltip": "Rank of the new acoustic LoRA. The starting companion is preserved separately in the exported delta."}),
            "window_frames": ("INT", {"default": 200, "min": 25, "max": 512, "tooltip": "Aligned training window at 25 frames per second. 200 frames = 8 seconds. Every pair must be at least this long."}),
            "head_learning_rate": ("FLOAT", {"default": .00001, "min": .0000001, "max": .01, "step": .000001, "tooltip": "Learning rate for head/joint mode."}),
            "acoustic_learning_rate": ("FLOAT", {"default": .0001, "min": .0000001, "max": .01, "step": .00001, "tooltip": "Learning rate for the acoustic LoRA and audio projection weights."}),
            "anchor_weight": ("FLOAT", {"default": .05, "min": 0, "max": 1, "step": .01, "tooltip": "Head/joint mode: preserve the starting encoder's source-token labels. This is a source-token anchor, not a generated-corpus regularizer."}),
            "source_replay": ("FLOAT", {"default": .1, "min": 0, "max": .5, "step": .01, "tooltip": "Fraction of updates reconstructing the source instead of the target, to limit drift. Set 0 for pure paired transformation."}),
            "accumulation": ("INT", {"default": 1, "min": 1, "max": 32, "tooltip": "Windows accumulated per optimizer update."}),
            "warmup_steps": ("INT", {"default": 20, "min": 0, "max": 10000, "tooltip": "Linear learning-rate warmup."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "control_after_generate": False, "tooltip": "Training initialization and sampling seed. Keep it unchanged when resuming."})},
            "optional": {
                "projection_learning_rate": ("FLOAT", {"default": .00002, "min": .0000001, "max": .01, "step": .000001, "tooltip": "Conditioned mode: audio input/output projection learning rate."}),
                "condition_learning_rate": ("FLOAT", {"default": .001, "min": .0000001, "max": .01, "step": .00001, "tooltip": "Conditioned mode: source projection learning rate. Warmup and cosine decay apply to all groups."}),
                "condition_dropout": ("FLOAT", {"default": .1, "min": 0, "max": 1, "step": .01, "tooltip": "Conditioned mode: probability of zeroing source latents during training, enabling source guidance."})}}

    def configure(self, **config):
        return (config,)


class FL_YuE2_AudioAdapterTrainer:
    CATEGORY = "FL YuE2/Audio Training"
    FUNCTION = "train"
    RETURN_TYPES = ("YUE2_AUDIO_ADAPTER",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "action": (["train", "use_saved"], {"tooltip": "Train a new/resumed adapter, or use an existing package without preparation."}),
            "output_name": ("STRING", {"default": "my_audio_adapter", "tooltip": "Run name. Use a new name for a new experiment; existing paired runs require explicit resume."}),
            "resume": ("STRING", {"default": "", "tooltip": "Enter resume.pt to continue this run, including after a cancelled preview."}),
            "selected_step": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 selects the latest trained checkpoint, not the baseline."}),
            "render_previews": ("BOOLEAN", {"default": True, "tooltip": "Render held-out source-to-target previews before training and at checkpoints."}),
            "preview_seconds": ("INT", {"default": 8, "min": 1, "max": 360, "tooltip": "Maximum held-out sample duration."}),
            "preview_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "tooltip": "Fixed noise seed for baseline and checkpoints."}),
            "sampling_steps": ("INT", {"default": 32, "min": 4, "max": 100, "tooltip": "Acoustic synthesis steps for previews."}),
            "validation_index": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Held-out pair to play. Index follows dataset order; changing it regenerates comparisons."})},
            "optional": {"assets": ("YUE2_TRAINING_ASSETS", {"lazy": True}), "dataset": ("YUE2_PREPARED_AUDIO_PAIRS", {"lazy": True}), "config": ("YUE2_AUDIO_TRAIN_CONFIG", {"lazy": True}), "condition_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 5, "step": .1, "tooltip": "Conditioned adapters: 1 uses source conditioning directly; 0 removes the source latent signal. Other values use two decoder passes for guidance."})},
            "hidden": {"unique_id": "UNIQUE_ID"}}

    def check_lazy_status(self, action, assets=None, dataset=None, config=None, **kwargs):
        return [key for key, value in (("assets", assets), ("dataset", dataset), ("config", config)) if value is None] if action == "train" else []

    def train(self, action, output_name, resume, selected_step, render_previews, preview_seconds, preview_seed, sampling_steps, validation_index,
              assets=None, dataset=None, config=None, unique_id=None, condition_scale=1.0):
        name = run_name(output_name)
        directory = output_root() / name
        path = directory / "run.json"
        settings = {"seed": preview_seed, "max_seconds": preview_seconds, "sampling_steps": sampling_steps, "validation_index": validation_index}
        if (config and config["mode"] == "conditioned") or (path.exists() and read_run(path)["config"]["mode"] == "conditioned"):
            settings["condition_scale"] = condition_scale
        preview_request = {"operation": "paired_preview", "run": str(path), **settings}
        if action == "train":
            if assets is None or dataset is None or config is None:
                raise ValueError("Connect training assets, prepared pairs, and audio adapter configuration")
            resume = str(contained(directory, resume)) if resume else ""
            if resume and render_previews and path.exists() and read_run(path).get("status") == "preview":
                run_worker(preview_request, unique_id)
            while True:
                run_worker({"operation": "paired_train", "assets": assets, "dataset": dataset, "config": config,
                            "run_directory": str(directory), "adapter_directory": str(adapter_root() / name), "resume": resume,
                            "pause_at_checkpoint": render_previews}, unique_id)
                run = read_run(path)
                if render_previews:
                    run_worker(preview_request, unique_id)
                if run["status"] == "complete":
                    break
                if run["status"] != "preview":
                    raise ValueError(f"Training stopped: {run['status']}")
                resume = str(directory / "resume.pt")
        elif action != "use_saved":
            raise ValueError("Unknown training action")
        run = read_run(path)
        if run["mode"] != "audio":
            raise ValueError("Select a paired-audio adapter run")
        candidates = [c for c in run["checkpoints"] if c["step"] == selected_step] if selected_step else run["checkpoints"][-1:]
        if not candidates:
            raise ValueError("No trained checkpoint at the selected step")
        if action == "use_saved" and render_previews and any(c.get("preview_settings") != settings or not c.get("preview") or not contained(directory, c["preview"]).exists() for c in [run["baseline"], *run["checkpoints"]]):
            run_worker(preview_request, unique_id)
        return {"ui": {"run": [name], "selected_step": [candidates[0]["step"]]}, "result": (candidates[0]["bundle"],)}


class FL_YuE2_AudioToAudio:
    CATEGORY = "FL YuE2"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls):
        choices = ["none"] + sorted(p.relative_to(adapter_root()).as_posix() for p in adapter_root().glob("*/step-*/manifest.json"))
        return {"required": {"audio": ("AUDIO",),
            "adapter_file": (choices, {"tooltip": "Paired adapter package under models/loras/YuE2_audio. A connected trainer package takes precedence."}),
            "style": ("STRING", {"default": "instrumental electronic music", "multiline": True, "tooltip": "Desired target style; use the training caption style for the first comparison."}),
            "lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional target lyrics."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "tooltip": "Acoustic noise seed."}),
            "sampling_steps": ("INT", {"default": 32, "min": 4, "max": 100, "tooltip": "Acoustic synthesis steps."})},
            "optional": {"adapter": ("YUE2_AUDIO_ADAPTER",), "condition_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 5, "step": .1, "tooltip": "Source latent guidance for conditioned adapters. 1 is the training baseline; values other than 1 require two decoder passes."})}, "hidden": {"unique_id": "UNIQUE_ID"}}

    def generate(self, audio, adapter_file, style, lyrics, seed, sampling_steps, adapter=None, unique_id=None, condition_scale=1.0):
        bundle = contained(adapter_root(), adapter if adapter else adapter_file)
        waveform = audio["waveform"]
        if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or not torch.isfinite(waveform).all():
            raise ValueError("Provide one finite mono/stereo source recording")
        root = output_root() / "inference"
        root.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex
        source, output = root / (tag + "-source.flac"), root / (tag + ".flac")
        sf.write(source, waveform[0].detach().cpu().T.numpy(), audio["sample_rate"], subtype="PCM_24")
        try:
            run_worker({"operation": "paired_infer", "source": str(source), "bundle": str(bundle), "style": style,
                        "lyrics": lyrics, "seed": seed, "sampling_steps": sampling_steps, "output": str(output), "condition_scale": condition_scale}, unique_id)
            samples, rate = sf.read(output, dtype="float32", always_2d=True)
            return ({"waveform": torch.from_numpy(samples.T.copy())[None], "sample_rate": rate},)
        finally:
            source.unlink(missing_ok=True)


AUDIO_NODES = {cls.__name__: cls for cls in (FL_YuE2_PairedDataset, FL_YuE2_PrepareAudioPairs, FL_YuE2_AudioAdapterConfig, FL_YuE2_AudioAdapterTrainer, FL_YuE2_AudioToAudio)}
AUDIO_NAMES = dict(zip(AUDIO_NODES, ("Paired Audio Dataset", "Prepare Audio Pairs", "Audio Adapter Config", "Audio Adapter Trainer", "Audio to Audio")))
