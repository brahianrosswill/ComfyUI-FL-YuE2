from . import runtime
from .score_editor import FL_YuE2_ScoreEditor


STYLE = "English, warm female vocal, melodic piano pop, gentle drums, acoustic guitar, uplifting, 90 BPM"
LYRICS = "[Verse]\nMorning light across the floor\nI hear the world outside my door\nEvery road begins with you\nEvery sky is turning blue\n\n[Chorus]\nTake me where the rivers run\nDancing underneath the sun\nHold this moment, let it stay\nWe will find another way"


class FL_YuE2_ModelLoader:
    CATEGORY = "FL YuE2"
    FUNCTION = "load"
    RETURN_TYPES = ("YUE2_MODEL", "YUE2_VAE")
    RETURN_NAMES = ("music_model", "audio_decoder")
    DESCRIPTION = "Load YuE2-3B and its stereo decoder. The first queued run downloads about 7.8 GB into models/yue2. Later runs work offline. NVIDIA BF16 GPU required."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"download_missing": ("BOOLEAN", {"default": True, "tooltip": "Download only missing YuE2 model files. Turn off for strictly offline loading."})}}

    def load(self, download_missing):
        return runtime.load_models(download_missing)


class FL_YuE2_Plan:
    CATEGORY = "FL YuE2"
    FUNCTION = "plan"
    RETURN_TYPES = ("YUE2_PLAN", "STRING")
    RETURN_NAMES = ("composition", "score_abc")
    DESCRIPTION = "Write a composition from musical style and optional lyrics. Full plans melody and chords; melody leaves harmony open; off skips the score. Instrumentals can use either planning mode with blank lyrics."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "music_model": ("YUE2_MODEL",),
            "style": ("STRING", {"multiline": True, "default": STYLE, "tooltip": "Describe language, genre, instruments, vocal character, mood and tempo."}),
            "lyrics": ("STRING", {"multiline": True, "default": LYRICS, "tooltip": "Words to sing, with [Verse], [Chorus], etc. Leave blank for instrumental music and describe it in style. Your planning choice still applies."}),
            "planning": (["full", "melody", "off"], {"default": "full", "tooltip": "full: melody + chords; melody: melody only; off: direct music generation."}),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 0x1FFFFFFFFFFFFF, "control_after_generate": True}),
            "max_score_tokens": ("INT", {"default": 4096, "min": 128, "max": 12000, "step": 128, "tooltip": "Safety limit for the written score. Does not set audio duration."}),
        }, "optional": {"score_abc": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional YuE2-compatible ABC score. Leave empty to compose one. Requires full or melody planning."})}}

    def plan(self, music_model, style, lyrics, planning, seed, max_score_tokens, score_abc=""):
        result = runtime.make_plan(music_model, style, lyrics, seed, planning, score_abc, max_score_tokens)
        return {"ui": {"text": [result.abc or "Direct generation — no symbolic score."]}, "result": (result, result.abc or "")}


class FL_YuE2_Render:
    CATEGORY = "FL YuE2"
    FUNCTION = "render"
    RETURN_TYPES = ("YUE2_LATENTS",)
    RETURN_NAMES = ("music_latents",)
    DESCRIPTION = "Turn a composition into music, then synthesize acoustic latents. Duration is a maximum: the model can finish earlier. Acoustic steps affect rendering quality, not song length."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "music_model": ("YUE2_MODEL",), "composition": ("YUE2_PLAN",),
            "max_duration": ("INT", {"default": 360, "min": 8, "max": 360, "step": 1, "tooltip": "Maximum seconds of generated music. Increase this if the ending is cut off."}),
            "acoustic_steps": ("INT", {"default": 32, "min": 1, "max": 64, "tooltip": "32 matches the released midpoint solver. Fewer steps are useful for quick tests."}),
        }, "optional": {
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Music-token sampling randomness. 0 uses greedy sampling."}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 100, "min": 1, "max": 1000}),
            "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 3.0, "step": 0.01}),
            "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01, "tooltip": "1.0 is the default for score-conditioned music. Direct mode's upstream default is 1.01. Values other than 1 use two generation branches."}),
        }}

    def render(self, music_model, composition, max_duration, acoustic_steps, temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2, guidance=1.0):
        latent, truncated, timing = runtime.render(music_model, composition, max_duration, temperature, top_p, top_k, repetition_penalty, guidance, acoustic_steps)
        status = "Duration limit reached — increase max_duration for a complete ending." if truncated else "Song generation complete."
        return {"ui": {"text": [status]}, "result": (latent,)}


class FL_YuE2_Decode:
    CATEGORY = "FL YuE2"
    FUNCTION = "decode"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    DESCRIPTION = "Decode to 48 kHz stereo audio. Connect to core Preview Audio or Save Audio. Tiles preserve the original decoder boundaries without crossfades."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_decoder": ("YUE2_VAE",), "music_latents": ("YUE2_LATENTS",),
                             "tile_frames": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64, "tooltip": "Smaller tiles reduce decoder VRAM. 1024 is the upstream default."})}}

    def decode(self, audio_decoder, music_latents, tile_frames):
        return (runtime.decode(audio_decoder, music_latents, tile_frames),)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (FL_YuE2_ModelLoader, FL_YuE2_Plan, FL_YuE2_Render, FL_YuE2_Decode, FL_YuE2_ScoreEditor)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_YuE2_ScoreEditor": "FL YuE2 · Piano Roll",
    "FL_YuE2_ModelLoader": "FL YuE2 · Load Models",
    "FL_YuE2_Plan": "FL YuE2 · Compose",
    "FL_YuE2_Render": "FL YuE2 · Render Music",
    "FL_YuE2_Decode": "FL YuE2 · Decode Audio",
}
