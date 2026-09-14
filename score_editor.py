import re
import math
import hashlib

from aiohttp import web
from server import PromptServer

from .abc_score import AbcError, KEYS, QUALITIES, DURATIONS, CHORD, parse, meter_value


DEFAULT_SCORE = '''X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Am
% verse
V: Vocal
"Am7"z16|"Dm7"z16|"G7"z16|"Cmaj7"z16|
V: Ins
E2A2c4B2A2G4|F2A2d4c2A2F4|G2B2d4c2B2A4|E2G2c4B2G2E4|
% outro
V: Vocal
"Fmaj7"z16|"Dm7"z16|"E7"z16|"Am7"z16|
V: Ins
F2A2c4A2G2F4|D2F2A4G2F2D4|E2^G2B4A2G2E4|A8z8|
'''


def inspect_score(text):
    if not isinstance(text, str) or len(text) > 200_000:
        raise AbcError("Supply ABC text of at most 200,000 characters.")
    text = text.replace("\r\n", "\n").strip() + "\n"
    score = parse(text)
    lines = text.splitlines()
    sections = []
    current = None
    grid_available = not any(line.startswith(("M:", "K:")) or "[K:" in line for line in lines[8:])
    for index in range(8, len(lines)):
        line = lines[index]
        if line.startswith("% "):
            current = {"name": line[2:], "vocal": [], "instrumental": []}
            sections.append(current)
        if index not in score.music_lines:
            continue
        if current is None:
            current = {"name": "section", "vocal": [], "instrumental": []}
            sections.append(current)
        voice = "vocal" if score.music_lines[index] == "Vocal" else "instrumental"
        for bar in line[:-1].split("|"):
            bar = bar.strip()
            rest = re.fullmatch(r"Z([2-4])?", bar)
            current[voice].extend(["Z"] * int(rest.group(1) or 1) if rest else [bar])
    sections = [section for section in sections if section["vocal"]]
    bars = len(score.voices["Vocal"].bars)
    seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
    instrumental = not score.voices["Vocal"].notes
    summary = f"Valid native ABC: {bars} bars, {score.bpm} BPM, approximately {seconds:.1f} seconds."
    summary += " Vocal part contains rests only." if instrumental else " Vocal melody present."
    if not grid_available:
        summary += " Key/meter changes: edit in the ABC tab."
    roll_sections = []
    start = 0
    for section in sections:
        roll_sections.append({"name": section["name"], "bars": len(section["vocal"]), "start": start})
        start += len(section["vocal"])
    roll = {"tracks": {name: [{"start": int(t * 256), "duration": int(d * 256), "pitch": pitch} for t, pitch, d in voice.notes]
                       for name, voice in score.voices.items()},
            "chords": [{"start": int(t * 256), "symbol": symbol} for t, symbol in score.voices["Vocal"].chords],
            "sections": roll_sections, "bar_ticks": int(score.voices["Vocal"].bars[0][1] * 256),
            "total_ticks": int(score.voices["Vocal"].time * 256)}
    return {"abc": text, "summary": summary, "bpm": score.bpm, "meter": lines[2][2:], "unit": lines[3][2:],
            "key": lines[7][2:], "seconds": seconds, "bars": bars, "instrumental": instrumental,
            "grid_available": grid_available, "sections": sections, "keys": list(KEYS), "qualities": list(QUALITIES), "roll": roll}


def build_score(data):
    def integer(value, label, minimum, maximum):
        if type(value) is not int or not minimum <= value <= maximum:
            raise AbcError(f"{label} must be an integer between {minimum} and {maximum}.")
        return value

    bpm = integer(data["bpm"], "Tempo", 1, 1000)
    key = data["key"]
    if key not in KEYS:
        raise AbcError("Choose a supported major or minor key.")
    meter = data["meter"]
    n, d = meter_value(meter)
    if 1024 * n % d:
        raise AbcError("Meter does not fit the editor grid.")
    bar_ticks = 1024 * n // d
    if bar_ticks > 32768:
        raise AbcError("The piano roll supports up to 128 quarter-note beats per bar.")
    roll = data["roll"]
    sections = roll["sections"]
    if not isinstance(sections, list) or not 1 <= len(sections) <= 256:
        raise AbcError("Supply 1–256 sections.")
    bars = sum(integer(section["bars"], "Section bars", 1, 256) for section in sections)
    if bars > 1024:
        raise AbcError("The score editor supports at most 1024 bars.")
    total = bars * bar_ticks
    tracks = {}
    grid = bar_ticks
    for name in ("Vocal", "Ins"):
        notes = roll["tracks"][name]
        if not isinstance(notes, list) or len(notes) > 20000:
            raise AbcError("Too many notes.")
        notes = sorted(notes, key=lambda note: note["start"])
        end = 0
        for note in notes:
            start = integer(note["start"], "Note position", 0, total - 1)
            duration = integer(note["duration"], "Note length", 1, total)
            integer(note["pitch"], "MIDI pitch", 0, 127)
            if start < end:
                raise AbcError(f"{name}: notes overlap. Each melody can play one note at a time.")
            end = start + duration
            if end > total:
                raise AbcError("A note extends past the end of the song.")
            grid = math.gcd(grid, math.gcd(start, duration))
        tracks[name] = notes
    chords = {}
    for chord in roll["chords"]:
        start = integer(chord["start"], "Chord position", 0, total - 1)
        if not isinstance(chord["symbol"], str) or not CHORD.fullmatch(chord["symbol"]):
            raise AbcError("Choose a supported chord.")
        if start in chords:
            raise AbcError("Only one chord can start at each position.")
        chords[start] = chord["symbol"]
        grid = math.gcd(grid, start)
    denominator = max(16, 1024 // (grid & -grid))
    unit = 1024 // denominator

    def pitch_text(pitch):
        names = ["=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B"]
        text = names[pitch % 12]
        octave = pitch // 12 - 1
        return text.lower() + "'" * (octave - 5) if octave >= 5 else text + "," * (4 - octave)

    def music_bar(name, index):
        start, end = index * bar_ticks, (index + 1) * bar_ticks
        notes = [note for note in tracks[name] if note["start"] < end and note["start"] + note["duration"] > start]
        boundaries = {start, end}
        for note in notes:
            boundaries.update((max(start, note["start"]), min(end, note["start"] + note["duration"])))
        if name == "Vocal":
            boundaries.update(t for t in chords if start <= t < end)
        boundaries = sorted(boundaries)
        parts = []
        for a, b in zip(boundaries, boundaries[1:]):
            if name == "Vocal" and a in chords:
                parts.append(f'"{chords[a]}"')
            note = next((note for note in notes if note["start"] <= a < note["start"] + note["duration"]), None)
            remaining = (b - a) // unit
            for length in sorted(DURATIONS, reverse=True):
                while remaining >= length:
                    remaining -= length
                    tied = note is not None and (remaining > 0 or b < note["start"] + note["duration"])
                    parts.append((pitch_text(note["pitch"]) if note else "z") + str(length) + ("-" if tied else ""))
        return "".join(parts)

    lines = ["X:1", "T:", f"M:{meter}", f"L:1/{denominator}", f"Q:1/4={bpm}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."', f"K:{key}"]
    offset = 0
    for section in sections:
        name = section["name"]
        if not isinstance(name, str) or len(name) > 100:
            raise AbcError("Section names must be text of at most 100 characters.")
        lines.append("% " + name.replace("\n", " ").replace("\r", " ").strip())
        for start in range(offset, offset + section["bars"], 4):
            for voice in ("Vocal", "Ins"):
                lines.extend((f"V: {voice}", "|".join(music_bar(voice, i) for i in range(start, min(start + 4, offset + section["bars"]))) + "|"))
        offset += section["bars"]
    return inspect_score("\n".join(lines))


class FL_YuE2_ScoreEditor:
    CATEGORY = "FL YuE2"
    FUNCTION = "score"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("score_abc",)
    OUTPUT_NODE = True
    DESCRIPTION = "Draw, move and resize melody notes in a piano roll, choose chords and preview the melody. Connect score_abc to Compose with full planning. Existing ABC scores can be imported or edited in the advanced view."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"score_abc": ("STRING", {"default": DEFAULT_SCORE, "multiline": True, "tooltip": "YuE2-compatible ABC notation edited by the piano roll. Connect the score output to Compose with full or melody planning."})},
                "optional": {"incoming_score_abc": ("STRING", {"forceInput": True, "tooltip": "Load a generated score for editing. Edits persist until the upstream score changes."}),
                             "source_score_hash": ("STRING", {"default": ""})}}

    def score(self, score_abc, incoming_score_abc=None, source_score_hash=""):
        incoming_hash = ""
        if incoming_score_abc is not None:
            incoming = inspect_score(incoming_score_abc)
            incoming_hash = hashlib.sha256(incoming["abc"].encode("utf-8")).hexdigest()
            if incoming_hash != source_score_hash:
                score_abc = incoming["abc"]
        result = inspect_score(score_abc)
        return {"ui": {"text": [result["summary"]], "score_abc": [result["abc"]], "source_score_hash": [incoming_hash]}, "result": (result["abc"],)}


async def validate_score(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise AbcError("Expected a JSON object containing score_abc.")
        return web.json_response(inspect_score(data.get("score_abc")))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


async def edit_score(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise AbcError("Expected a score object.")
        return web.json_response(build_score(data))
    except (ValueError, KeyError, TypeError) as error:
        return web.json_response({"error": str(error)}, status=400)


if hasattr(PromptServer, "instance"):
    PromptServer.instance.routes.post("/fl_yue2/score/validate")(validate_score)
    PromptServer.instance.routes.post("/fl_yue2/score/build")(edit_score)
