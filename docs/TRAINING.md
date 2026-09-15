# AR LoRA training

Training supports AR song-generation LoRAs only. NAR/joint training and acoustic-corpus building have been removed. YuE2 still uses its acoustic model and VAE internally to render audio.

## Models

Training Models exposes named tokenizer/regularizer selections and `download_missing`. When queued, it resolves the following assets inside ComfyUI's registered model folders and downloads missing files only if enabled:

- `models/yue2/YuE2-3B` and `YuE2-Vae`: pinned public base weights.
- `models/yue2/MERT-v2-FullSong`: pinned MERT encoder and supporting files.
- `models/yue2/training_assets/tokenizer_head_joint_v4.pt`: [Mothersuperior v4 tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).
- `models/yue2/training_assets/minted_regularizer_pack.pt`: [public regularizer pack](https://huggingface.co/datasets/Mothersuperior/yue2-minted-corpus).
- The tokenizer's pretrained acoustic companion is downloaded and converted internally to `models/loras/YuE2/pretrained/`. It is used for rendering, never trained by these nodes.

Downloads support partial-file resume and checksum verification. Existing valid files work offline. The optional Demucs/MMS alignment weights download only when lyric alignment is requested. Install Python training dependencies from [requirements-training.txt](../requirements-training.txt); workers do not install packages.

## Dataset and training

Dataset Maker and Gemini accept audio folders relative to ComfyUI's input directory, or explicitly supplied absolute audio folders. Preparation cache names are relative to `output/yue2_training`. No model-directory widgets are required.

Each recording needs `.caption.txt` and `.lyrics.txt` sidecars; empty lyrics mean no intelligible vocals. Enter a Google API key in the Gemini Music Captioner node; machine environment keys are not used. The captioner uploads the selected audio to Google. The key is passed to the worker over stdin rather than written to job files. Normal ComfyUI widgets are saved in workflows/history, so clear the key before sharing them. Generated captions are automatically accepted, including existing captions. Use Load captions and Save changes for optional corrections; manual review is not required. Incomplete transcriptions retry shorter intervals with overlapping context; unresolved intervals report their track and times.

Prepare Dataset extracts frozen MERT features and predicts semantic tokens with the selected head. English lyric alignment is optional; disable it and use cursor weight zero for uncertain or processed vocal samples. AR training uses the token regularizer pack, with no VAE-latent preparation or neighbor arrays.

Train Config exposes rank, learning rate, regularizer fraction, cursor weight, sequence budget, steps, checkpoint interval, cosine horizon, warmup, accumulation and seed. With `action=train` and blank `resume`, every queue starts fresh and overwrites the named run, including its old checkpoints and previews. Change `output_name` to keep an earlier experiment. Existing saved AR runs remain available through `use_saved`; changing to v4 does not retokenize or retrain them automatically.

## Checkpoints and playback

Adapters are saved under `models/loras/YuE2/<run>/`; metrics, previews and complete optimizer/RNG resume state are under `output/yue2_training/<run>/`. Resume a matching run with `resume.pt`; explicit resume preserves existing results and still requires matching data, assets and training settings.

With `render_previews` enabled, the trainer first saves its step-0 resume state and renders a baseline before any optimizer updates. It then renders a sample at every `save_every` checkpoint before continuing training. Training releases GPU memory for inference, then resumes the saved optimizer and random state. Rendering and model reloads add time at each checkpoint; finished samples remain playable while training continues. After interruption, set `resume` to `resume.pt`. Disable previews for uninterrupted training.

A separate inference progress bar shows model loading, generated duration, synthesis steps, decoding chunks and saving. The training counter and loss chart remain unchanged during previews. Baseline - Step 0 is the first playable sample and cannot be selected as an exported LoRA. It uses the starting model and the same acoustic companion, prompt, seed and duration as checkpoint previews. Matching previews are reused on resume; changing preview settings regenerates the comparisons. An older run can generate its missing baseline from its starting model.

Play a validation sample to activate the seek bar below the samples. Click or drag to jump through the audio; elapsed time and duration are displayed. Seeking preserves whether playback is paused or running, and also works for the baseline.

`Use` selects a checkpoint and switches to `use_saved`, which skips preparation and training. Load LoRA exposes AR selection and strength only; connected saved runs retain their original pretrained acoustic companion. New exports record that companion in their metadata for standalone loading.

Start with [training_studio.json](../example_workflows/training_studio.json). For a two-step pipeline check, use [training_smoke.json](../example_workflows/training_smoke.json) with your own reviewed audio. The smoke preset explicitly truncates sequences and is not a quality-training preset.

See `TRAINING_VALIDATION.md` for measured tests and quality limitations.

The captioner runs up to `concurrent_requests` recordings at once (default 3, range 1-8). Set 1 for sequential operation or lower it if Google returns rate-limit errors. Excerpts within each song remain sequential. Completed songs are saved as they finish, with the manifest kept in filename order. Cancellation or a failed song stops new work; in-flight API requests may finish before cancellation takes effect.
