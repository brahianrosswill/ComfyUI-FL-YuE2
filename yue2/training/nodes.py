from pathlib import Path
import uuid

import folder_paths
from comfy_api.latest import io
from safetensors import safe_open

from .. import adapters
from ..downloads import resolve, MODELS
from . import downloads as training_downloads
from .data import dataset, read_run, run_name, contained, fingerprint, audio_files
from .service import output_root, run_worker


CATEGORY = "FL YuE2/Training"


def lora_root():
    return Path(folder_paths.get_folder_paths("loras")[0]) / "YuE2"


def selected(root, name):
    if name == "none":
        return ""
    path = contained(root, name)
    if not path.is_file():
        raise ValueError(f"Missing asset: {path.name}")
    return str(path)


class FL_YuE2_TrainingModels:
    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("YUE2_TRAINING_ASSETS",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tokenizer_head": (["tokenizer_head_joint_v4.pt"], {"tooltip": "Encoder head used to convert MERT audio features into AR training tokens. Saved under models/yue2/training_assets."}),
            "regularizer": (["minted_regularizer_pack.pt"], {"tooltip": "Precomputed generated-music examples mixed with your recordings to help preserve the base model."}),
            "download_missing": ("BOOLEAN", {"default": True, "tooltip": "Download missing training models into ComfyUI model folders when queued. Disable for offline loading."}),
        }}

    def load(self, tokenizer_head, regularizer, download_missing=True):
        if tokenizer_head != "tokenizer_head_joint_v4.pt" or regularizer != "minted_regularizer_pack.pt":
            raise ValueError("Unknown AR training asset selection")
        model = resolve("YuE2-3B", download_missing)
        resolve("YuE2-Vae", download_missing)
        mert = resolve("MERT-v2-FullSong", download_missing)
        head = training_downloads.asset(tokenizer_head, download_missing)
        pack = training_downloads.asset(regularizer, download_missing)
        acoustic = training_downloads.acoustic_adapter(model, download_missing)
        return ({"model": str(model), "head": str(head), "mert": str(mert),
                 "model_revision": MODELS["YuE2-3B"], "mert_revision": MODELS["MERT-v2-FullSong"],
                 "regularizer": str(pack), "initial_nar": str(acoustic),
                 "download_missing": download_missing},)


class FL_YuE2_GeminiMusicCaptioner:
    CATEGORY = CATEGORY
    FUNCTION = "caption"
    RETURN_TYPES = ("YUE2_CAPTIONS",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_directory": ("STRING", {"default": "", "tooltip": "Audio folder, relative to ComfyUI/input or an absolute folder. Recordings are sent to Google for captioning."}),
                             "model": (["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro"], {"tooltip": "Gemini model that listens to each recording. Uses the Google API key entered on this node; API usage may incur charges."}),
                             "task": (["both", "style", "lyrics"], {"tooltip": "Generate style descriptions, transcribe lyrics, or do both. Review the resulting text before training."}), "replace_existing": ("BOOLEAN", {"default": False, "tooltip": "Regenerate existing captions instead of reusing saved results."}),
                             "instructions": ("STRING", {"multiline": True, "default": "", "tooltip": "Additional directions for Gemini, such as how to describe the genre or handle unclear vocals."}),
                             "api_key": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False, "tooltip": "Google Gemini API key for this run. Machine environment keys are not used. Clear this field before sharing a workflow; normal ComfyUI widgets are saved with the workflow."}),
                             "concurrent_requests": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Recordings captioned at the same time. 1 is sequential. Reduce this if Google returns quota/rate-limit errors. Song excerpts stay in order within each recording."})}, "optional": {"test_random": ("BOOLEAN", {"default": False})}, "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, audio_directory, test_random=False, replace_existing=False, **kwargs):
        if test_random or replace_existing:
            return float("nan")
        files = audio_files(Path(folder_paths.get_input_directory()) / audio_directory)
        return fingerprint([(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files])

    def caption(self, audio_directory, model, task, replace_existing, instructions, unique_id=None, api_key="", concurrent_requests=3, test_random=False):
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("Enter a Google API key on the Gemini Music Captioner node")
        root = (Path(folder_paths.get_input_directory()) / audio_directory).resolve(strict=True)
        name = uuid.uuid4().hex[:24] if test_random else fingerprint(str(root))[:24]
        path = output_root() / "captions" / name / "manifest.json"
        result = run_worker({"operation": "caption", "directory": str(root), "model": model, "task": task,
                             "replace_existing": replace_existing, "instructions": instructions, "test_random": test_random, "concurrent_requests": concurrent_requests, "output": str(path)}, unique_id, api_key=api_key)
        return {"ui": {"caption_test" if test_random else "captions": [name]}, "result": (result,)}


class FL_YuE2_DatasetMaker:
    CATEGORY = CATEGORY
    FUNCTION = "make"
    RETURN_TYPES = ("YUE2_DATASET",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_directory": ("STRING", {"default": "", "tooltip": "Folder containing recordings and caption sidecars, relative to ComfyUI/input or an absolute folder."}), "trigger": ("STRING", {"default": "", "tooltip": "Text added to each training style caption. Include the same text when generating music with the LoRA."}),
                             "default_style": ("STRING", {"default": "", "multiline": True, "tooltip": "Fallback style description for recordings without their own style caption."}),
                             "validation_fraction": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "Fraction of recordings held out for validation instead of training. 0.15 reserves about 15%."}),
                             "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "tooltip": "Random seed for the train/validation split. Keep fixed to compare runs on the same split."})}, "optional": {"captions": ("YUE2_CAPTIONS",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def make(self, audio_directory, trigger, default_style, validation_fraction, seed, captions=None):
        directory = Path(folder_paths.get_input_directory()) / audio_directory
        return (dataset(directory, trigger, default_style, validation_fraction, seed, captions),)


class FL_YuE2_PrepareDataset:
    CATEGORY = CATEGORY
    FUNCTION = "prepare"
    RETURN_TYPES = ("YUE2_PREPARED_DATASET",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dataset": ("YUE2_DATASET",), "assets": ("YUE2_TRAINING_ASSETS",),
                             "align_lyrics": ("BOOLEAN", {"default": False, "tooltip": "Separate vocals and align reviewed lyrics to the recording. Requires additional alignment models and preparation time."}),
                             "cache_directory": ("STRING", {"default": "prepared", "tooltip": "Folder inside output/yue2_training"}),
                             "score_planning": (["off", "melody", "full"], {"default": "off", "tooltip": "Train from reviewed .abc.txt sidecars. Full keeps melody and chords; melody expects a melody-only score."}),
                             "transcribe_missing_scores": ("BOOLEAN", {"default": False, "tooltip": "Use pinned SheetSage2 to create missing ABC scores during preparation. SheetSage2 is CC BY-NC 4.0."})},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    def prepare(self, dataset, assets, score_planning, transcribe_missing_scores, align_lyrics, cache_directory, unique_id=None):
        if align_lyrics:
            assets = {**assets, "alignment_models": str(training_downloads.alignment(assets["download_missing"]))}
        if score_planning != "off" and transcribe_missing_scores:
            assets = {**assets, "sheetsage": training_downloads.sheetsage(assets["download_missing"])}
        return (run_worker({"operation": "prepare", "dataset": dataset, "assets": assets,
                            "align": align_lyrics, "score_planning": score_planning,
                            "cache_directory": str(contained(output_root(), cache_directory))}, unique_id),)


class FL_YuE2_TrainConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_YuE2_TrainConfig", display_name="FL YuE2 · Train Config", category=CATEGORY,
            inputs=[io.Int.Input("rank", default=64, min=1, max=128, tooltip="LoRA rank controls adapter capacity and size. Higher ranks use more training memory and can learn more detail."),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=0.01, step=1e-6, round=False, extra_dict={"precision": 7}, tooltip="Optimizer update size. Higher values learn faster but can destabilize training or overfit."),
                io.Float.Input("generated_fraction", default=0.5, min=0.01, max=0.99, step=0.01, extra_dict={"precision": 2}, tooltip="Probability of training on a regularizer example instead of your recordings. 0.5 gives each source an equal chance."),
                io.Float.Input("cursor_weight", default=0.08, min=0, max=1, step=0.01, extra_dict={"precision": 2}, tooltip="Weight of the auxiliary lyric-position loss. Set to 0 to disable it, especially for instrumental data."),
                io.Int.Input("sequence_tokens", default=12288, min=256, max=24576, tooltip="Maximum training sequence length in tokens. Full songs need larger values and more VRAM."),
                io.Boolean.Input("allow_truncation", default=False, tooltip="Allow sequences longer than sequence_tokens to be cut short. Disabled makes oversized examples fail instead of silently losing their endings."),
                io.Int.Input("steps", default=1600, min=1, max=100000, tooltip="Total optimizer steps to reach, including steps already completed when resuming."), io.Int.Input("save_every", default=200, min=1, max=5000, tooltip="Save a LoRA and resumable training checkpoint every this many optimizer steps. The final step is also saved."),
                io.Int.Input("schedule_steps", default=3000, min=1, max=100000, tooltip="Length of the learning-rate decay schedule. Keep unchanged when resuming to preserve the schedule."), io.Int.Input("warmup_steps", default=50, min=0, max=10000, tooltip="Initial optimizer steps over which the learning rate rises to its configured value."),
                io.Int.Input("accumulation", default=2, min=1, max=64, tooltip="Training examples accumulated before each optimizer update. Higher values increase work per step without batching them all in VRAM."), io.Int.Input("seed", default=42, min=0, max=2147483647, tooltip="Random seed for training example selection and LoRA initialization. Keep fixed for reproducible comparisons."),
                io.Boolean.Input("train_acoustic", default=False, optional=True, tooltip="Also train a token-to-audio acoustic companion from your recordings. Adds VAE target preparation and decoder training; source audio is not needed for generation. Checkpoints automatically include both adapters.")],
            outputs=[io.Custom("YUE2_TRAIN_CONFIG").Output()])

    @classmethod
    def execute(cls, **kwargs):
        if not kwargs.get("train_acoustic"):
            kwargs.pop("train_acoustic", None)
        return io.NodeOutput({"mode": "ar", **kwargs})


class FL_YuE2_JointTrainConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_YuE2_JointTrainConfig", display_name="FL YuE2 · Joint Score Train Config", category=CATEGORY,
            inputs=[io.Int.Input("rank", default=32, min=1, max=128, tooltip="LoRA rank for both AR and NAR transformer projections."),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=0.01, step=1e-6, round=False, extra_dict={"precision": 7}),
                io.Float.Input("weight_decay", default=1e-4, min=0, max=0.1, step=1e-5, round=False, extra_dict={"precision": 6}),
                io.Float.Input("ar_kl_weight", default=0.2, min=0, max=2, step=0.01, extra_dict={"precision": 2}, tooltip="KL(base || adapter) weight used to preserve the base AR distribution."),
                io.Float.Input("abc_dropout", default=0.5, min=0, max=1, step=0.05, extra_dict={"precision": 2}, tooltip="Probability of training an example in direct mode without its score."),
                io.Float.Input("ar_lr_multiplier", default=1.0, min=0.01, max=2, step=0.05, extra_dict={"precision": 2}),
                io.Int.Input("acoustic_window_frames", default=1500, min=64, max=3000, step=64, tooltip="Random NAR flow-matching window. YuE2 uses 25 frames per second."),
                io.Combo.Input("nar_start", options=["base", "community_v4"], default="base", tooltip="AI Toolkit starts from base NAR; community_v4 folds the released acoustic adapter first."),
                io.Int.Input("sequence_tokens", default=24576, min=256, max=24576),
                io.Boolean.Input("allow_truncation", default=False),
                io.Int.Input("steps", default=3000, min=1, max=100000),
                io.Int.Input("save_every", default=200, min=1, max=5000),
                io.Int.Input("accumulation", default=1, min=1, max=64),
                io.Int.Input("seed", default=42, min=0, max=2147483647)],
            outputs=[io.Custom("YUE2_TRAIN_CONFIG").Output()])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput({"mode": "ar", "recipe": "ai_toolkit_joint_v1", **kwargs})


class FL_YuE2_LoRATrainer:
    CATEGORY = CATEGORY
    FUNCTION = "train"
    RETURN_TYPES = ("YUE2_ADAPTER",)
    RETURN_NAMES = ("adapter",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "action": (["train", "use_saved"], {"tooltip": "Use saved selects/previews checkpoints without running preparation or training."}),
            "output_name": ("STRING", {"default": "my_song_lora", "tooltip": "Run folder inside output/yue2_training. Training with blank resume overwrites this run and its saved checkpoints/previews. Change the name to keep the old run."}),
            "resume": ("STRING", {"default": "", "tooltip": "Resume checkpoint path relative to this run folder. Leave blank to train from scratch and overwrite existing results with this output name."}),
            "selected_step": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 selects the latest saved checkpoint."}),
            "render_previews": ("BOOLEAN", {"default": True, "tooltip": "Render a step-0 baseline and a sample at each save_every checkpoint, then resume training. Releases training VRAM during inference; adds rendering and reload time. Also renders missing samples for saved runs."}),
            "preview_style": ("STRING", {"default": "instrumental piano", "multiline": True, "tooltip": "Style prompt used for every checkpoint sample. Include your dataset trigger for a useful comparison."}),
            "preview_lyrics": ("STRING", {"default": "", "multiline": True, "tooltip": "Lyrics for checkpoint samples. Leave blank for instrumental previews."}),
            "preview_seed": ("INT", {"default": 42, "control_after_generate": False, "tooltip": "Fixed generation seed shared by checkpoint samples so differences are easier to compare."}),
            "preview_seconds": ("INT", {"default": 30, "min": 8, "max": 360, "tooltip": "Maximum duration in seconds for each checkpoint sample. Longer samples take more time to render."}),
        }, "optional": {
            "assets": ("YUE2_TRAINING_ASSETS", {"lazy": True}),
            "dataset": ("YUE2_PREPARED_DATASET", {"lazy": True}),
            "config": ("YUE2_TRAIN_CONFIG", {"lazy": True}),
            "preview_ar_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "AR adapter strength for validation audio only. 0 disables the trained AR adapter; 1 applies it fully. Does not affect training."}),
            "preview_nar_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "Acoustic companion strength for validation audio, including the step-0 baseline. Scales the entire NAR adapter, including its pretrained component. Does not affect training."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, action, **kwargs):
        return float("nan") if action == "train" else "use_saved"

    def check_lazy_status(self, action, **kwargs):
        return [key for key in ("assets", "dataset", "config") if key in kwargs and kwargs[key] is None] if action == "train" else []

    def train(self, action, output_name, resume, selected_step, render_previews, preview_style, preview_lyrics,
              preview_seed, preview_seconds, assets=None, dataset=None, config=None, unique_id=None,
              preview_ar_strength=1.0, preview_nar_strength=1.0):
        name = run_name(output_name)
        directory = output_root() / name
        path = directory / "run.json"
        settings = {"style": preview_style, "lyrics": preview_lyrics, "seed": preview_seed, "max_seconds": preview_seconds,
                    "ar_strength": preview_ar_strength, "nar_strength": preview_nar_strength}
        if action == "train":
            if assets is None or dataset is None or config is None:
                raise ValueError("Connect training assets, prepared dataset, and config to train")
            if resume:
                resume = str(contained(directory, Path(resume)))
            if resume and render_previews and path.is_file():
                baseline = read_run(path).get("baseline", {})
                if baseline.get("preview_settings") != settings or not baseline.get("preview") or not contained(directory, baseline["preview"]).is_file():
                    run_worker({"operation": "preview", "run": str(path), **settings}, unique_id)
            while True:
                run_worker({"operation": "train", "assets": assets, "dataset": dataset, "config": config,
                            "run_directory": str(directory), "adapter_directory": str(lora_root() / name), "resume": resume,
                            "pause_at_checkpoint": render_previews}, unique_id)
                run = read_run(path)
                if run["status"] == "complete":
                    break
                if run["status"] != "preview":
                    raise RuntimeError(f"Training stopped with status {run['status']}; resume from resume.pt")
                run_worker({"operation": "preview", "run": str(path), **settings}, unique_id)
                resume = str(directory / "resume.pt")
        elif action != "use_saved":
            raise ValueError("Unknown trainer action")
        if not path.is_file():
            raise ValueError(f"No saved run named {name}; train it first or select an existing run")
        run = read_run(path)
        if run["mode"] != "ar":
            raise ValueError("Select an AR LoRA run")
        candidates = [c for c in run["checkpoints"] if c["step"] == selected_step] if selected_step else run["checkpoints"][-1:]
        if not candidates:
            raise ValueError("No checkpoint at the selected step")
        chosen = candidates[0]
        if render_previews and any(c.get("preview_settings") != settings or not c.get("preview") or
                                   not contained(directory, c["preview"]).is_file() for c in [run.get("baseline", {}), *run["checkpoints"]]):
            run_worker({"operation": "preview", "run": str(path), **settings}, unique_id)
        descriptor = {"ar": chosen["adapter"], "nar": chosen.get("acoustic_adapter", run["assets"].get("initial_nar", ""))}
        return {"ui": {"run": [name], "selected_step": [chosen["step"]]}, "result": (descriptor,)}


class FL_YuE2_LoadLoRA:
    CATEGORY = "FL YuE2"
    FUNCTION = "load"
    RETURN_TYPES = ("YUE2_MODEL",)

    @classmethod
    def INPUT_TYPES(cls):
        names = ["none"]
        for path in sorted(lora_root().rglob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as file:
                if (file.metadata() or {}).get("branch") == "ar":
                    names.append(path.relative_to(lora_root()).as_posix())
        return {"required": {"music_model": ("YUE2_MODEL",), "ar_adapter": (names, {"tooltip": "Saved AR LoRA under models/loras/YuE2. A connected trainer adapter takes precedence over this selection."}),
                             "ar_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "tooltip": "Adapter strength: 0 disables its effect and 1 applies the trained weights at full strength."})},
                "optional": {"adapter": ("YUE2_ADAPTER",),
                             "nar_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "Acoustic companion strength: 0 disables the entire NAR adapter, including its pretrained component; 1 applies it fully. Does not change AR strength."})}}

    def load(self, music_model, ar_adapter, ar_strength, adapter=None, nar_strength=1.0):
        path = str(contained(lora_root(), adapter["ar"])) if adapter and adapter.get("ar") else selected(lora_root(), ar_adapter)
        acoustic = adapter.get("nar", "") if adapter else ""
        if path:
            with safe_open(path, framework="pt", device="cpu") as file:
                metadata = file.metadata() or {}
                if metadata.get("branch") != "ar":
                    raise ValueError("Select an AR LoRA adapter")
                if not adapter:
                    acoustic = metadata.get("acoustic_adapter", "")
        paths = [path]
        if acoustic:
            paths.append(str(contained(lora_root(), acoustic)))
        return (adapters.patch_music(music_model, paths, ar_strength, nar_strength),)


TRAINING_NODES = {c.__name__: c for c in (FL_YuE2_TrainingModels, FL_YuE2_GeminiMusicCaptioner, FL_YuE2_DatasetMaker,
    FL_YuE2_PrepareDataset, FL_YuE2_TrainConfig, FL_YuE2_JointTrainConfig, FL_YuE2_LoRATrainer, FL_YuE2_LoadLoRA)}
