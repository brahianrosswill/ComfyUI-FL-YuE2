# AR LoRA training

Training supports the original AR recipe and an AI Toolkit-style joint score recipe. The joint recipe trains AR and NAR transformer LoRAs together from score, semantic-token, and VAE targets. Normal generation still needs only style and lyrics; it does not require a source recording.

## Models

Training Models exposes named tokenizer/regularizer selections and `download_missing`. When queued, it resolves the following assets inside ComfyUI's registered model folders and downloads missing files only if enabled:

- `models/yue2/YuE2-3B` and `YuE2-Vae`: pinned public base weights.
- `models/yue2/MERT-v2-FullSong`: pinned MERT encoder and supporting files.
- `models/yue2/training_assets/tokenizer_head_joint_v4.pt`: [Mothersuperior v4 tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).
- `models/yue2/training_assets/minted_regularizer_pack.pt`: [public regularizer pack](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus).
- The tokenizer's pretrained acoustic companion is downloaded and converted internally to `models/loras/YuE2/pretrained/`. It is used for rendering and initializes optional acoustic training. Exported acoustic companions include both the starting adapter and the learned update.
- Joint score preparation can download pinned SheetSage2 weights, source, and AI Toolkit compatibility code when `transcribe_missing_scores` is enabled. It is never downloaded at startup. SheetSage2 is licensed CC BY-NC 4.0.

Downloads support partial-file resume and checksum verification. Existing valid files work offline. The optional Demucs/MMS alignment weights download only when lyric alignment is requested. Install Python training dependencies from [requirements-training.txt](../requirements-training.txt); workers do not install packages.

## Dataset and training

Dataset Maker and Gemini accept audio folders relative to ComfyUI's input directory, or explicitly supplied absolute audio folders. Preparation cache names are relative to `output/yue2_training`. No model-directory widgets are required.

Each recording needs `.caption.txt` and `.lyrics.txt` sidecars; empty lyrics mean no intelligible vocals. Joint score training additionally accepts a reviewed `.abc.txt` sidecar. Prepare Dataset can fill missing scores with pinned SheetSage2 when full or melody planning and `transcribe_missing_scores` are selected. Enter a Google API key in the Gemini Music Captioner node; machine environment keys are not used. The captioner uploads the selected audio to Google. The key is passed to the worker over stdin rather than written to job files. Normal ComfyUI widgets are saved in workflows/history, so clear the key before sharing them. Generated captions are automatically accepted, including existing captions. Use Load saved captions and Save caption edits for optional corrections; manual review is not required. Incomplete transcriptions retry shorter intervals with overlapping context; unresolved intervals report their track and times.

The captioner's two-row panel contains a system-prompt editor, folder progress, and an audio/text preview. Your directions supplement the built-in transcription instructions. **Process folder** runs only this captioner, using `replace_existing` and `concurrent_requests`; the main ComfyUI Queue still runs the connected workflow. **Test random track** captions one random recording with the same prompt and model, saving a separate preview under `output/yue2_training/captions` without changing dataset sidecars. Listen and seek through the source audio beside its style and lyrics. Panel actions use local widget values; use the main Queue if inputs are connected. Tests and folder runs share ComfyUI's queue and interruption controls.

Prepare Dataset extracts frozen MERT features and predicts semantic tokens with the selected head. English lyric alignment is optional; disable it and use cursor weight zero for uncertain or processed vocal samples. AR training uses the token regularizer pack. With acoustic training enabled, the training worker additionally caches VAE posterior-mean targets under `output/yue2_training/acoustic_targets`; these targets are not needed at inference.

Train Config exposes rank, learning rate, regularizer fraction, cursor weight, sequence budget, steps, checkpoint interval, cosine horizon, warmup, accumulation and seed. With `action=train` and blank `resume`, every queue starts fresh and overwrites the named run, including its old checkpoints and previews. Change `output_name` to keep an earlier experiment. Existing saved AR runs remain available through `use_saved`; changing to v4 does not retokenize or retrain them automatically.

## Joint score recipe

Joint Score Train Config implements `ai_toolkit_joint_v1`: rank 32, constant 0.0001 learning rate, 0.0001 weight decay, full-song AR cross-entropy, `KL(base || LoRA)` weight 0.2, 50% score dropout, and sigmoid-timestep NAR flow matching on random windows. It uses real recordings rather than the minted replay pack and starts NAR from the base model by default. Select `community_v4` only for an explicit comparison.

Preparation records the planning mode and score hashes. A joint run cannot resume from a legacy checkpoint or from a differently prepared dataset. AR and NAR remain separate exported files so existing Load LoRA workflows continue to work. Joint previews use the trained planning mode and generate a new score before music tokens.

The NAR window defaults to 1,500 frames (60 seconds) and can be configured from 64 to 3,000 frames. Reduce `acoustic_window_frames` for smaller cards. The effective window is also capped by the recording length and model context remaining after the prompt and score.

## Optional acoustic companion (experimental)

`train_acoustic` defaults to off, preserving AR-only training. When enabled, each optimizer step also trains a NAR LoRA and audio input/output projections on a window of a real recording. Its semantic tokens are the conditioning; its VAE latents are the loss target, never an extra decoder input. AR gradients, regularizer sampling and random state remain separate. The MERT head and VAE remain fixed.

Acoustic windows are at most 512 frames (20.48 seconds). The companion uses the configured LoRA rank, accumulation, warmup and decay schedule, with learning rates 0.00005 for LoRA and 0.00002 for audio projections. It has its own optimizer and resume state. The minted pack contains no waveform targets, so acoustic updates use real recordings only; AR regularization is unchanged. Acoustic validation reports held-out flow loss at noise times 0.2, 0.5 and 0.8.

The [controlled benchmark](ACOUSTIC_BENCHMARK.md) improved held-out reconstruction but did not consistently improve the requested character in text-only generation. This option therefore remains experimental and off by default. Semantic compression still limits reconstruction fidelity. It does not add paired source-audio injection or train score planning. Enabling it changes the trainable weights and requires a fresh run rather than resuming an AR-only checkpoint.

## Checkpoints and playback

Adapters are saved under `models/loras/YuE2/<run>/`; metrics, previews and complete optimizer/RNG resume state are under `output/yue2_training/<run>/`. Resume a matching run with `resume.pt`; explicit resume preserves existing results and still requires matching data, assets and training settings.

With `render_previews` enabled, the trainer first saves its step-0 resume state and renders a baseline before any optimizer updates. It then renders a sample at every `save_every` checkpoint before continuing training. Training releases GPU memory for inference, then resumes the saved optimizer and random state. Rendering and model reloads add time at each checkpoint; finished samples remain playable while training continues. After interruption, set `resume` to `resume.pt`. Disable previews for uninterrupted training.

A separate inference progress bar shows model loading, generated duration, synthesis steps, decoding chunks and saving. The training counter and loss chart remain unchanged during previews. Baseline - Step 0 is the first playable sample and cannot be selected as an exported LoRA. It uses the configured NAR starting point with the same prompt, seed and duration as checkpoint previews: the base NAR for a default joint run, or the pretrained community companion for legacy acoustic training and joint runs that select `community_v4`. Acoustic-enabled checkpoints use their own learned companion. Matching previews are reused on resume; changing preview settings regenerates the comparisons. An older run can generate its missing baseline from its starting model.

Play a validation sample to activate the seek bar below the samples. Click or drag to jump through the audio; elapsed time and duration are displayed. Seeking preserves whether playback is paused or running, and also works for the baseline.

`Use` selects a checkpoint and switches to `use_saved`, which skips preparation and training. Load LoRA exposes independent AR and NAR strengths; connected saved runs load the selected checkpoint's learned acoustic companion when present, otherwise the original pretrained companion. AR exports record the companion's relative path in metadata for standalone loading. Keep `step-NNNNNN.safetensors` and `step-NNNNNN-nar.safetensors` together under the same run folder when copying acoustic-enabled checkpoints.

Start with [training_studio.json](../example_workflows/training_studio.json). For a two-step pipeline check, use [training_smoke.json](../example_workflows/training_smoke.json) with your own reviewed audio. The smoke preset explicitly truncates sequences and is not a quality-training preset.

See `TRAINING_VALIDATION.md` for measured tests and quality limitations.

The captioner runs up to `concurrent_requests` recordings at once (default 3, range 1-8). Set 1 for sequential operation or lower it if Google returns rate-limit errors. Excerpts within each song remain sequential. Completed songs are saved as they finish, with the manifest kept in filename order. Cancellation or a failed song stops new work; in-flight API requests may finish before cancellation takes effect.

Load LoRA exposes independent `ar_strength` and `nar_strength` controls, both defaulting to 1. The NAR control scales the entire acoustic companion, including its pretrained component; 0 disables that companion. It applies to both a selected AR file's companion and a connected trainer adapter.

The LoRA Trainer exposes `preview_ar_strength` and `preview_nar_strength` under its validation preview settings (0–2, default 1). These affect preview audio only, not optimization or exported weights. The NAR setting also applies to the pretrained companion in the step-0 baseline. Both strengths are recorded with each sample and included in its cache key.
