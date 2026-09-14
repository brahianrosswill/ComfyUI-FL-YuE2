"""Explicit Gemini audio requests and editable dataset sidecars."""
import json
import os
import time
import tempfile
from pathlib import Path

import soundfile as sf

from ..downloads import digest
from .data import audio_files, sidecar, read_json, write_json, fingerprint


PROMPT = """Listen to the entire recording. Return a concise music style caption describing only audible genre, instruments, vocal character, language, mood, groove, arrangement and production. Do not identify the artist. Tempo and key may be omitted if uncertain.
Transcribe all audible lyrics verbatim in their original language and order. Preserve every repetition. Do not summarize, translate, invent words, use ellipses for repeated sections, or look up lyrics. Use [Verse], [Chorus], [Bridge], [Intro], [Outro] when justified. Mark uncertain passages in the separate uncertainty field. For instrumental music use empty lyrics and instrumental=true. If the response cannot cover the complete song, set complete=false. Timestamps are approximate review aids, not alignment targets."""

SCHEMA = {"type": "object", "properties": {
    "style": {"type": "string"}, "lyrics": {"type": "string"}, "instrumental": {"type": "boolean"},
    "uncertainty": {"type": "string"}, "complete": {"type": "boolean"}},
    "required": ["style", "lyrics", "instrumental", "uncertainty", "complete"], "additionalProperties": False}


def validate_response(data):
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise ValueError("Gemini returned an incomplete caption schema")
    if any(not isinstance(data[k], str) for k in ("style", "lyrics", "uncertainty")) or any(type(data[k]) is not bool for k in ("instrumental", "complete")):
        raise ValueError("Gemini returned invalid caption field types")
    if not data["complete"]:
        raise ValueError("Gemini did not transcribe the full recording; shorten the input into reviewed song sections and retry")
    if not data["style"].strip() or (data["instrumental"] and data["lyrics"].strip()) or (not data["instrumental"] and not data["lyrics"].strip()):
        raise ValueError("Gemini returned inconsistent instrumental/lyric content")
    return data



def listen(client, audio, request, prompt, emit, cancelled):
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(audio))
        deadline = time.monotonic() + 300
        while str(getattr(uploaded.state, "name", uploaded.state)) == "PROCESSING":
            cancelled()
            if time.monotonic() > deadline:
                raise TimeoutError("Gemini audio processing timed out")
            time.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        if str(getattr(uploaded.state, "name", uploaded.state)) == "FAILED":
            raise ValueError("Gemini could not process the audio")
        for attempt in range(3):
            cancelled()
            try:
                interaction = client.interactions.create(model=request["model"], store=False,
                    input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}, {"type": "text", "text": prompt}],
                    response_format=SCHEMA, timeout=300)
                break
            except Exception as error:
                # Only transient API failures are retried; refusals/schema failures are surfaced.
                if getattr(error, "status_code", None) not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"Gemini request failed ({type(error).__name__}); check the selected model and credentials") from None
                for _ in range(2 ** (attempt + 1)):
                    cancelled()
                    time.sleep(1)
        cancelled()
        return json.loads(interaction.output_text)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                emit({"type": "status", "message": "Temporary Google file cleanup failed; remove it in Google AI Studio"})


def caption(request, emit, cancelled, client=None):
    if client is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in the ComfyUI environment, then restart ComfyUI")
        client = genai.Client(api_key=key)
    files = audio_files(request["directory"])
    task = request["task"]
    if task not in {"both", "style", "lyrics"}:
        raise ValueError("Unknown caption task")
    result = {"version": 1, "directory": str(files[0].parent), "songs": []}
    for index, audio in enumerate(files):
        cancelled()
        emit({"type": "status", "message": f"Captioning {index + 1}/{len(files)}: {audio.name}"})
        metadata_path = audio.with_suffix(".caption.json")
        paths = {kind: sidecar(audio, "caption" if kind == "style" else "lyrics") for kind in ("style", "lyrics")}
        requested = [k for k in paths if task == "both" or task == k]
        needed = [k for k in requested if request["replace_existing"] or not paths[k].exists()]
        audio_hash = digest(audio)
        if metadata_path.exists():
            previous = read_json(metadata_path)
            if previous.get("audio_sha256") != audio_hash and not request["replace_existing"]:
                raise ValueError(f"{audio.name}: audio changed since captioning; explicitly regenerate or supply reviewed sidecars")
        if needed:
            prompt = PROMPT + "\n" + request.get("instructions", "")
            data = listen(client, audio, request, prompt, emit, cancelled)
            segments = []
            if isinstance(data, dict) and data.get("complete") is False:
                style = data.get("style", "")
                lyrics, uncertainty = [], []
                with sf.SoundFile(audio) as source, tempfile.TemporaryDirectory(prefix="yue2-caption-") as temporary:
                    rate, frames = source.samplerate, len(source)
                    window = min(180 * rate, (frames + 1) // 2)
                    pending = [(start, min(frames, start + window)) for start in range(0, frames, window)]
                    while pending:
                        cancelled()
                        start, end = pending.pop(0)
                        left, right = max(0, start - 3 * rate), min(frames, end + 3 * rate)
                        source.seek(left)
                        chunk = Path(temporary) / "segment.flac"
                        sf.write(chunk, source.read(right - left), rate)
                        segment_prompt = prompt + f"\nThis is an overlapping excerpt. Transcribe only words STARTING from {(start-left)/rate:.3f} seconds up to (excluding) {(end-left)/rate:.3f} seconds within this excerpt. Audio outside this interval is context only. Preserve genuine repetitions inside the interval. Mark boundary uncertainty separately. Here complete means you covered this interval, not the original full song. Unintelligible speech belongs in uncertainty; do not invent its words."
                        emit({"type": "status", "message": f"Transcribing {audio.name}: {start/rate:.1f}-{end/rate:.1f}s"})
                        part = listen(client, chunk, request, segment_prompt, emit, cancelled)
                        if isinstance(part, dict) and part.get("complete") is False:
                            if end - start <= 15 * rate:
                                raise ValueError(f"{audio.name}: Gemini could not complete {start/rate:.1f}-{end/rate:.1f}s; review this interval before retrying")
                            middle = (start + end) // 2
                            pending[0:0] = [(start, middle), (middle, end)]
                            continue
                        part = validate_response(part)
                        lyrics.append(part["lyrics"].strip())
                        if part["uncertainty"]:
                            uncertainty.append(f"{start/rate:.1f}-{end/rate:.1f}s: {part['uncertainty']}")
                        segments.append({"start": start/rate, "end": end/rate, "lyrics": part["lyrics"]})
                text = "\n".join(value for value in lyrics if value)
                data = {"style": style, "lyrics": text, "instrumental": not text, "uncertainty": "\n".join(uncertainty), "complete": True}
            data = validate_response(data)
            for kind in needed:
                temporary = paths[kind].with_suffix(".tmp")
                temporary.write_text(data[kind].strip(), encoding="utf-8")
                os.replace(temporary, paths[kind])
            write_json(metadata_path, {"model": request["model"], "audio_sha256": audio_hash,
                       "prompt_hash": fingerprint(prompt), "uncertainty": data["uncertainty"], "reviewed": False,
                       "instrumental": data["instrumental"], "segments": segments})
        metadata = read_json(metadata_path) if metadata_path.exists() else {"reviewed": True}
        result["songs"].append({"name": audio.stem, "audio": str(audio), "metadata": str(metadata_path),
                                "style": str(paths["style"]), "lyrics": str(paths["lyrics"]), **metadata})
        write_json(request["output"], result)
        emit({"type": "progress", "step": index + 1, "max_steps": len(files)})
    return request["output"]
