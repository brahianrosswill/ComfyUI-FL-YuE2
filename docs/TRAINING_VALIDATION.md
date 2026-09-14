# Training validation

Version 0.2.0 was checked on Windows, Python 3.12, Torch 2.11 and an RTX PRO 6000 Blackwell. Other hardware and minimum VRAM requirements remain unverified.

- Full AR training and saved-checkpoint inference completed on real recordings with the v4 tokenizer and public regularizer pack.
- A two-step GPU run rendered and played checkpoint 1 before step 2, then rendered checkpoint 2. All 392 final adapter tensors exactly matched an uninterrupted run.
- Resume restores optimizer and Python/NumPy/Torch/CUDA random state. Preview failure or cancellation does not advance training.
- Browser checks verified play/pause, playback across checkpoint refreshes, checkpoint selection, and native widget tooltips.
- Asset tests cover checksum verification, interrupted transfers, offline reuse and path containment. Installed training assets also resolved with networking disabled.
- Automated tests cover training gradients, adapter export, caption review, incomplete transcription handling, saved selection, interleaved previews and run-file locking.

These are correctness and integration checks, not evidence that a particular checkpoint sounds good. Captions still need review, and learned musical quality needs listening tests. Gemini transcription may miss or misinterpret processed vocals.
