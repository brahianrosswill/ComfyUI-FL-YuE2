# YuE2 LoRA training assessment

Checked September 10, 2026. Current upstream `main` is still `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`. This is a feasibility assessment, not an implemented trainer or a validated training recipe.

YuE2 has useful model components for adaptation, but the public release does not include an end-to-end finetuning package. The familiar `finetune/` guide belongs to the archived YuE-v1 branch and targets `YuE-s1-7B-anneal-en-cot`. Its xcodec data and LoRA checkpoints are not interchangeable with YuE2.

Sources: [current source tree](https://github.com/multimodal-art-projection/YuE/tree/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e), [v1 finetuning guide](https://github.com/multimodal-art-projection/YuE/blob/YuE-v1/finetune/README.md).

## What is available

| Component | Evidence and implications |
|---|---|
| AR model forward and loss | `src/yue2/modeling_yue2.py:492` accepts labels and computes shifted cross-entropy. A tiny CPU test produced finite loss and attention gradients with `use_cache=False`; NAR attention received no gradient. This demonstrates the AR backward path only, not successful LoRA adaptation. |
| Acoustic VAE encoder | `src/yue2/modeling_vae.py:491` encodes 48 kHz stereo to continuous acoustic latents. The installed checkpoint contains 218 encoder tensor entries. Our Comfy inference loader currently loads only the decoder. |
| Semantic audio tokenizer | The encoder docstring explicitly calls this a separate, unreleased component. Neither the current GitHub tree nor the checked YuE2-3B/VAE asset lists provides it. VAE latents, text tokens, MERT embeddings, and v1 xcodec codes cannot substitute for its discrete IDs. |
| NAR velocity implementation | `modeling_yue2.py:609` exposes the flow velocity computation, but is decorated with `no_grad` and supplies no training loss/optimizer/data recipe. A differentiable training path and verified noise/time/target conventions are still needed. |
| Gradient checkpointing | The model advertises support, but the inspected backbone loop does not implement checkpoint calls. Actual operation must be implemented/verified before offering it as a memory option. |
| Training package | No training/LoRA/finetuning paths in the current recursive tree. No YuE2 training data recipe, validated data-size requirement, or training VRAM minimum found in the checked release documentation. The technical report is listed as forthcoming. |

Sources: [AR and NAR implementation](https://github.com/multimodal-art-projection/YuE/blob/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e/src/yue2/modeling_yue2.py), [VAE implementation](https://github.com/multimodal-art-projection/YuE/blob/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e/src/yue2/modeling_vae.py), [VAE model card](https://huggingface.co/m-a-p/YuE2-Vae), [generation model assets](https://huggingface.co/m-a-p/YuE2-3B/tree/main).

## Practical training scopes

These are engineering proposals inferred from the released architecture, not upstream-supported recipes.

| Goal | Dataset needed | Feasibility |
|---|---|---|
| Score composition LoRA | Style, lyrics or instrumental intent, reviewed native YuE2 ABC score; full versus melody mode recorded | Best first implementation. No audio tokenizer needed. Learns score patterns, not a singer's acoustic identity or a mastering sound. |
| Semantic music LoRA | Style, lyrics, optional matching score, genuine YuE2 semantic tokens | Can build a trainer for verified pre-tokenized examples. Arbitrary recordings remain blocked on the matching encoder/codebook. |
| Acoustic rendering LoRA | Conditioning text/score/semantic tokens paired with aligned VAE latents from the same recording | Requires a new NAR training path and exact pairing/preprocessing. This is the relevant research direction for timbre/production adaptation, but not validated. |
| Text-only acoustic experiment | Real audio VAE latents and text, without semantic conditioning | The upstream codec-dropout/text-only attention path suggests an experiment is possible. It does not establish that the adapter transfers correctly to normal semantic-conditioned generation. Treat separately from supported training. |

Generated YuE2 artifacts contain semantic tokens and latents and can support trainer plumbing tests or synthetic-data experiments. Generated tokens paired with an unrelated real recording are invalid supervision. Synthetic examples do not establish learning from a user's recording collection.

For a recording dataset, collect audio, descriptive style tags, accurate lyrics for each excerpt (empty for instrumentals), song identity and split, excerpt boundaries, and optional reviewed native ABC. Keep validation split by song so adjacent excerpts do not leak between training and validation. Semantic/acoustic caches must record model revisions, token representation, frame count, and alignment. Confirm encoder normalization, VAE posterior sampling, and time alignment rather than guessing them.

No official minimum minutes, song count, or singing-voice recipe was found. An initial experiment could use short 10–30-second excerpts and batch size one, but these are development choices, not quality guarantees. The existing approximately 96 GB BF16 GPU is a reasonable development target; actual peak VRAM and step time need measurement. The published 24 GB figure is for inference. Rank 16 or 32, gradient accumulation, and verified activation checkpointing are candidate starting settings, not YuE2 defaults. Long sequences and the 184,704-way vocabulary make activations/logits a major memory consideration even with a small adapter.

## Local FL pack comparison

- **VoxCPM:** `nodes/dataset_maker_node.py`, `nodes/train_config_v2_node.py`, `nodes/lora_trainer_node.py`, `modules/trainer_v2.py`, `web/voxcpm_training.js`. The node calls the training loop within Comfy execution; the inspected path is not a separate worker process. It provides JSONL dataset creation, rank/alpha/learning-rate controls, local websocket progress, loss plots, interrupt checks, safetensors adapters, and fixed-seed checkpoint audio previews. Checkpoint exports shown here contain adapter weights and configuration; they do not by themselves provide optimizer/scheduler/RNG state for exact resume.
- **Qwen3-TTS:** `nodes/training_ui.py` provides an all-in-one preparation/training interface. Its validation reloads the saved model, a useful pattern for checking real export/inference compatibility. Its speech-specific targets are not YuE2 supervision.
- **HeartMuLa:** `fl_nodes/lora_trainer.py:349` and `experiments/train_lora.py:226` generate pseudo-random codec IDs from an audio-energy seed. This checks execution plumbing but does not encode musical content. Do not copy this path into a real YuE2 trainer. This finding is about these local files, not a claim about every HeartMuLa implementation.
- **SongGen:** contains model/trainer and vendored training code, but its tokenizer and checkpoint format are not YuE2-compatible. Reuse applicable interface conventions, not its token data.

## Proposed node implementation

1. **Prepare Training Dataset:** validate the selected supported mode, file containment, captions/lyrics/scores, token identities, duration/alignment, and train/validation split. Produce a local manifest and reusable caches. Missing semantic tokens must be an explicit unsupported path rather than fabricated data.
2. **Train LoRA:** a VoxCPM-style panel with basic controls, optional advanced configuration, live loss/learning-rate/progress, checkpoint list, cancel, resume, and fixed-seed validation samples. Keep dependencies optional. An isolated local worker is the proposed YuE2 execution design to separate training autograd/model lifetime from Comfy inference; it is not a description of the current VoxCPM implementation. Release inference allocations before training and restore usable generation afterward.
3. **Load YuE2 LoRA:** read adapters from `models/loras/YuE2/<run>/`, validate base revision and target shapes, clone the music model's ModelPatcher and apply through Comfy's existing LoRA patch machinery. The current custom `YUE2_MODEL` socket cannot directly plug into core Load LoRA. Reuse the core patch engine with a narrow model-specific adapter; use existing Render/Decode/Preview/Save nodes downstream.

Store only adapter tensors in the shareable safetensors export, with target names/rank/alpha/base revision/training mode metadata. Store resumable optimizer, scheduler, step, random generator, and sampler state separately. Avoid network logging, telemetry, package installation at startup, or core Comfy modifications. Any future model downloads should fetch only explicitly requested training artifacts on execution.

Candidate first AR targets are the exact `model.layers.*.self_attn.{q,k,v,o}_proj` layers; test MLP targets only after an attention baseline. NAR targets use separate `nar_self_attn` and `nar_mlp` paths. An AR-only loss will not train these NAR modules. Do not assume attaching LoRA to every projection trains every generation stage. The existing Comfy inference implementation uses cached execution and optimized kernels; verify backward/patch contracts before sharing any of those paths with a trainer.

## Acceptance gates

Before exposing a usable training node: overfit a small genuine dataset; verify finite nonzero adapter gradients and unchanged base weights; measure peak memory; save/reload the adapter in the installed Comfy generation path; confirm strength zero restores baseline; compare held-out fixed-seed score/audio outputs against the base model; test cancel/cleanup and resume equivalence. Falling training loss alone is insufficient.

First deliverable should be a validated score-only trainer or verified pre-tokenized AR trainer. A folder-of-recordings workflow needs the official semantic tokenizer, matching precomputed targets, or a separately validated acoustic-only research approach. Resolve that data path before promising voice/style LoRA training.

The current source is Apache 2.0; model weights are separately marked CC BY-NC 4.0 in the [model card](https://huggingface.co/m-a-p/YuE2-3B). The archived v1 license should not be substituted for the current weights' license.
