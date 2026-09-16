# Text-to-music acoustic companion benchmark

Tested 2026-09-15 on Windows, Python 3.12, PyTorch 2.11 and an RTX PRO 6000 Blackwell Max-Q (96 GB).

**Result:** adapting the acoustic decoder improved held-out token-conditioned reconstruction, but did not consistently improve low-pass character in text-only generation. `train_acoustic` remains experimental and off by default.

## Controlled comparison

The original AR trainer was run twice: AR-only, and the same AR training plus an acoustic companion. Eight distinct 8-second music excerpts were processed with a sixth-order 1,800 Hz low-pass filter and 0.65 gain. Six songs trained the models; two entire songs were held out. The recordings are local fixtures and are not distributed.

Both runs used rank 16, 300 steps, accumulation 2, AR learning rate 0.00005, warmup 20, decay horizon 300, seed 42, 50% minted AR regularization and no cursor loss. The sequence budget was 1,024 with explicit truncation for the longer regularizer examples. This is a short controlled benchmark, not a full-song quality-training preset.

The acoustic run additionally trained NAR LoRA at 0.00005 and audio projections at 0.00002. It used target semantic tokens as conditioning and target VAE latents only as supervision. Neither training nor generation supplied an extra source-latent condition. The MERT head and VAE stayed fixed.

At steps 100, 200 and 300, **all 392 exported AR tensors were bit-identical between the two runs**. The AR losses and validation losses matched. Acoustic parameters had an independent optimizer and random stream.

## Held-out reconstruction

Both decoders received the same held-out recording's semantic tokens, caption and noise seed. Audio was rendered with 32 midpoint steps. Error is the Frobenius norm of the difference between stereo STFT magnitudes divided by the target magnitude norm (2,048-sample FFT, 512-sample hop), averaged over the two held-out songs. The VAE's short output-length difference was trimmed before comparison.

| Checkpoint | AR-only companion | Learned companion | Error reduction |
| --- | ---: | ---: | ---: |
| 100 | 0.8660 | 0.6511 | 24.8% |
| 200 | 0.9385 | 0.6400 | 31.8% |
| 300 | 0.9495 | 0.6315 | 33.5% |

This measures token-conditioned reconstruction, not free-generation quality or source-audio copying. At step 300, individual errors were 1.1200 to 0.5509 for one held-out song and 0.7791 to 0.7122 for the other. Improvement was uneven.

## Text-only generation

The final checkpoints generated 8-second clips from `instrumental deep dub music, warm low-pass filtered sound`, blank lyrics, planning off, and seeds 42, 123 and 2026. Sampling used temperature 1, top-p 0.95, top-k 100, repetition penalty 1.2, guidance 1.01 and 32 acoustic steps. **Generated semantic tokens were identical for each matched seed.** Only the decoder changed.

For this deliberately low-pass target, lower relative power above 3 kHz indicates closer adherence to the requested frequency cutoff. It is not a general music-quality score.

| Seed | AR-only power above 3 kHz | Learned companion | AR-only RMS | Learned RMS |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.75738% | 1.56239% | 0.2186 | 0.1340 |
| 123 | 0.00001338% | 0.00000681% | 0.0790 | 0.0926 |
| 2026 | 0.02921% | 0.12133% | 0.1359 | 0.1690 |

Two of three seeds worsened on this frequency-cutoff measure. The outputs did not collapse to silence, but better reconstruction did not reliably transfer to better text-only style adherence. No claim of improved arrangement, vocals or general perceptual quality is supported by this experiment.

## Runtime and integration

- Training segments including checkpoint evaluation took about 158 seconds AR-only versus 331 seconds with the companion; feature preparation, preview rendering, reloads and export overhead are not all included in those segment timers.
- Peak allocated training memory was 7.73 versus 8.00 GiB for these short examples.
- Both runs completed through the original ComfyUI trainer with baseline and checkpoint previews, automatically resuming between preview workers.
- Fresh-process seed-42 rendering matched the saved previews within FLAC quantization (maximum absolute sample difference 1.2e-7).
- The normal ComfyUI Load LoRA / Compose / Render / Decode graph also completed successfully and produced playable 48 kHz stereo audio. Its waveform was not identical to the worker preview at the same seed; cross-process frontend/worker reproducibility was not established. All quantitative A/B results above use the same worker environment.
- Tests cover acoustic gradient isolation, unchanged AR updates, exact resumed AR/acoustic weights, merged companion export, and standalone companion resolution through Load LoRA. The pack suite passed 115 tests.

Local runs are `text_acoustic_bench_ar` and `text_acoustic_bench_joint`. Raw measurements and matched audio are under `output/yue2_text_acoustic_benchmark/`. Listen to the checkpoints before selecting a production adapter. Broader datasets and longer text-only evaluations are needed before recommending this as a default.
