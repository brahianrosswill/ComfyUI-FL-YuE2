import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.htdemucs import HTDemucs

from .data import fingerprint
from .prepare import resample


def align(song, folder, assets, emit, cancelled):
    path = folder / f"cursor-{fingerprint(song['lyrics'])[:16]}.npy"
    if path.exists():
        return str(path)
    if not assets.get("alignment_models"):
        raise ValueError("Install the Demucs/MMS alignment models explicitly before enabling lyric alignment")
    words, offset = [], 0
    for line in song["lyrics"].split("\n"):
        if not re.fullmatch(r"\s*\[.*\]\s*", line):
            for match in re.finditer(r"\S+", line):
                word = re.sub(r"[^a-z']", "", match.group().lower().replace("’", "'"))
                if word:
                    words.append((offset + match.start(), offset + match.end(), word))
        offset += len(line) + 1
    if not words:
        raise ValueError("No English words available for lyric alignment")
    cancelled()
    emit({"type": "status", "message": "Separating vocals for lyric alignment"})
    with torch.serialization.safe_globals([HTDemucs, Fraction, np.dtype, np.core.multiarray.scalar, type(np.dtype("float64"))]):
        separator = get_model("htdemucs", repo=Path(assets["alignment_models"]) / "demucs").to("cuda").eval()
    for parameter in separator.parameters():
        parameter.requires_grad_(False)
    audio, rate = sf.read(song["audio"], dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    wav = torch.tensor(resample(audio, rate, separator.samplerate).T[None], device="cuda")
    reference = wav.mean(1, keepdim=True)
    mean, std = reference.mean(), reference.std().clamp_min(1e-8)
    stems = apply_model(separator, (wav - mean) / std, device="cuda", shifts=0, split=True) * std + mean
    vocals = stems[0, separator.sources.index("vocals")].mean(0).detach().cpu().numpy()
    source_rate = separator.samplerate
    del separator, stems, wav
    cancelled()
    bundle = torchaudio.pipelines.MMS_FA
    # The training-assets installer resolves the exact checkpoint before this call.
    checkpoint = Path(assets["alignment_models"]) / "mms_alignment.pt"
    if not checkpoint.is_file():
        raise ValueError("Install MMS alignment weights before preparing lyric timings")
    model = bundle.get_model(with_star=False, dl_kwargs={"model_dir": str(checkpoint.parent), "file_name": checkpoint.name}).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    labels = {label: i for i, label in enumerate(bundle.get_labels(star=None))}
    tokens = [[labels[c] for c in word if c in labels] for _, _, word in words]
    if not all(tokens):
        raise ValueError("Lyrics include unsupported alignment characters")
    audio16 = resample(vocals, source_rate, 16000)
    emission = model(torch.tensor(audio16[None], device="cuda"))[0].log_softmax(-1)
    alignment, scores = torchaudio.functional.forced_align(emission, torch.tensor([[v for word in tokens for v in word]], device="cuda"), blank=0)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0].exp())
    seconds_per_frame = len(audio16) / 16000 / emission.shape[1]
    rows, position = [], 0
    for (start, end, _), token in zip(words, tokens):
        segment = spans[position:position + len(token)]
        position += len(token)
        rows.append([segment[0].start * seconds_per_frame, segment[-1].end * seconds_per_frame,
                     float(np.mean([float(s.score) for s in segment])), start, end])
    np.save(path, np.asarray(rows, dtype=np.float32))
    return str(path)
