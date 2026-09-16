# Paired audio adapter training

This experimental path learns **source audio → target audio**. It conditions acoustic synthesis on tokens extracted from the source; it does not generate an AR song or train a score planner. The existing AR LoRA workflow remains separate.

## Workflow

`Paired Audio Dataset → Prepare Audio Pairs → Audio Adapter Trainer → Audio to Audio → Preview Audio`

Connect **Training Models** to preparation and training, **Audio Adapter Config** to training, and **Load Audio** to inference. Training Models resolves the existing YuE2, MERT, VAE, tokenizer-head and acoustic-companion assets in ComfyUI model folders. Its download switch controls missing asset downloads. The AR regularizer selected there is not consumed by paired training.

An [example workflow](../example_workflows/audio_adapter_training.json) is included. Set its audio folders and choose your own source file before running; audio and trained adapters are not bundled.

Put source and target recordings in separate folders with matching filename stems. WAV, FLAC and MP3 are accepted. Both sides must be aligned, mono or stereo, and have matching durations. Relative folders resolve under ComfyUI `input`. Preparation converts audio to 48 kHz stereo, extracts source MERT features at 25 Hz, and encodes target audio using the VAE posterior mean. Source and target windows always share the same frame offset.

Optional target sidecars:

- `song.caption.txt`: target style; otherwise use the node's default style.
- `song.lyrics.txt`: target lyrics; otherwise empty.
- `song.song.txt`: original-song identifier. Give every excerpt of the same song the same identifier so it cannot leak across the training/validation split.

Use genuinely different source/target pairs to train a transformation. Examples include aligned dry/wet processing, an instrument stem/full mix, or an intentionally processed version of the source. The system does not align unrelated songs, isolate stems, or construct paired data automatically. Each recording must be at least as long as `window_frames / 25` seconds.

## Training modes

| Mode | Updated weights |
| --- | --- |
| Conditioned | New NAR LoRA, audio input/output projections, and four source-latent projections; source token head stays fixed |
| Acoustic | New NAR LoRA plus audio input/output projections |
| Head | Source MERT-to-token head, with gradients through the frozen AR and acoustic model |
| Joint | Both the head and acoustic adapter |

AR weights, MERT and VAE stay fixed. The starting acoustic companion is retained in the final exported delta. Head/joint training uses straight-through token selection. `anchor_weight` anchors the head to the starting source-token labels; `source_replay` sometimes reconstructs the source instead of the target. These are explicit regularizers, **not** Kytra's full minted-corpus replay objective. Set source replay to zero for a pure paired transformation.

Validation holds out entire song groups. The chart compares target flow loss using the correct source against the same target/noise using a different source. Lower loss alone does not prove that the desired transformation sounds good: compare the audio as well.

With previews enabled, training saves and plays an untrained **step-0 baseline**, then pauses at saved checkpoints for inference before continuing. Source, target, baseline and checkpoint samples share the trainer carousel and seek bar. Preview seed and validation recording stay fixed for a fair comparison.

## Continuous source conditioning

`conditioned` follows the decoder approach in [Kytra's hum-to-song training script](https://huggingface.co/Mothersuperior/YuE2-hum-to-song/blob/cd323af53ebf61d613fb3d7717e7c8b9bb67f643/scripts/train_hum.py): zero-initialized 64-channel source projections enter decoder layers 0, 7, 14 and 21. Source VAE posterior-mean latents retain detail that semantic tokens can discard. No hum extraction or other new preprocessing is applied.

Baseline settings: rank 64, LoRA learning rate 0.00005, input/output projection rate 0.00002, condition rate 0.001, accumulation 2, warmup 50, condition dropout 0.1, and source replay 0. Conditioned mode uses cosine decay to 20% of each learning rate. Validation evaluates noise times 0.2, 0.5 and 0.8 with correct, wrong and zero source conditioning; `run.json` records the best saved validation step. These losses are not directly comparable to older modes' single-time validation.

Our paired transformation retains source semantic tokens during both training and inference. Kytra's hum-to-song setup instead trains with target tokens and generates tokens through AR at inference. This is an adaptation for source-to-source tasks, not a reproduction of her full hum pipeline. Wrong/zero-condition validation keeps semantic tokens fixed to isolate the continuous source channel.

`condition_scale=1` uses the trained condition directly. Other values blend zero-source and conditioned decoder velocities and require two decoder passes. Zeroing this channel still retains the source semantic tokens. Start at 1 when evaluating fidelity.

## Save, resume and inference

Paired runs live under `output/yue2_training/<output_name>`. Unlike AR fresh-run overwrite, paired training requires a new name or explicit `resume.pt` when a run already exists. Resume restores trainable parameters, optimizer and random states and verifies dataset, configuration and base assets. Cancellation preserves the most recent saved checkpoint; unsaved updates are lost.

Portable inference packages live under `models/loras/YuE2_audio/<output_name>/step-000100/`:

- `manifest.json`: relative weight filenames, hashes, base identity and feature revisions.
- `head.safetensors`: trained or starting source head.
- `acoustic.safetensors`: combined starting/trained acoustic adapter, when present.
- `conditioning.safetensors`: source projections for conditioned mode. Required alongside the other package files.

Copy the whole checkpoint folder, not a single weight file. **Audio to Audio** accepts the trainer output or a package chosen from its model-folder list. Refresh the node definitions after copying packages. Inference checks the base model and package hashes and uses the original source recording plus target style/lyrics. It does not need the training dataset or optimizer checkpoint. Required base models must already be available; inference itself does not download them.

## Practical limits

This is a generative source-conditioned adapter, not a sample-exact audio effect or a general guarantee of style transfer. MERT/token compression may discard details needed for some transformations. Start with short aligned pairs and a held-out test, then increase data diversity and duration. Training windows are limited to the source head's 512 frames (20.48 seconds); longer recordings are sampled in aligned windows. Full-recording source features are normalized before windowing in both training and inference.

The current training and feature preparation paths require CUDA and the optional `requirements-training.txt` dependencies. Workers own their GPU allocations and exit between training and preview phases. No ComfyUI core execution or model classes are modified.

## Development validation (2026-09-15)

Eight distinct 8-second music excerpts were paired with aligned targets processed by a sixth-order 1,800 Hz low-pass filter and 0.65 gain. Six songs trained the adapter; two were held out. These recordings are local test fixtures, not distributed with the repository.

The acoustic test used rank 8, 200-frame windows, 40 updates, learning rate 0.0003, two warmup steps and no source replay. Previews used seed 42 and 16 synthesis steps.

The test machine used an NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB), Windows and PyTorch 2.11. Other GPU sizes/backends were not validated by this test.

| Check | Result |
| --- | --- |
| Held-out correct-source flow loss | 0.926 baseline → 0.812 at step 40 |
| Held-out wrong-source flow loss | 0.971 baseline → 0.835 at step 40 |
| Preview power above 3 kHz | 2.183% baseline → 0.0000485% trained |
| Preview normalized target magnitude-spectrum error | 0.829 baseline → 0.885 trained; did not improve |
| Fresh-process exported inference | Sample-identical to the step-40 preview with the same source, caption and seed |
| Training GPU allocation peak | About 7.2 GiB acoustic; 7.8 GiB joint, for this short-window test |

The test demonstrates a learned paired transformation and continued dependence on source audio, not general style transfer or waveform fidelity. The output suppressed the intended high-frequency band without collapsing to silence, but its amplitude and detailed spectrum were not an exact match to the target.

Head-only and joint GPU smoke tests verified finite gradients and changes to the intended weight groups. Head-only changed 103 head tensors and left the acoustic companion unchanged; joint changed both groups. A joint run was cancelled during its baseline preview, resumed from `resume.pt`, and completed through step 6 with checkpoint previews. Ordinary acoustic training also resumed across separate workers at steps 0 and 20.

Tests cover paired-data alignment and song splits, changed-source detection, short-tail feature extraction, training/inference flow parity, source gradients, combined-adapter export, stable training seeds and normalized/padded head inference. Conditioned tests also cover zero-initialized baseline parity, training/inference parity and gradients, zero-source guidance, checkpoint integrity, and source alignment across inference chunks. The full pack suite passed 113 tests in the development environment.

The existing ComfyUI workflow was modified and saved in place. Frontend checks confirmed source/target/baseline/checkpoint playback, click-to-seek, inference progress events, and stable seeds after restart and reload.
