import re

from aiohttp import web
from server import PromptServer

from .service import output_root
from .data import read_json, write_json, read_run, write_text, contained, fingerprint, run_name


def caption_manifest(value):
    if not re.fullmatch(r"[0-9a-f]{24}", value):
        raise ValueError("Invalid caption manifest")
    return output_root() / "captions" / value / "manifest.json"


async def get_captions(request):
    try:
        result = read_json(caption_manifest(request.match_info["identifier"]))
        rows = []
        for song in result["songs"]:
            values = {}
            for key in ("style", "lyrics"):
                path = contained(result["directory"], song[key])
                values[key] = path.read_text(encoding="utf-8") if path.exists() else ""
            rows.append({"name": song["name"], "reviewed": song.get("reviewed", False), "uncertainty": song.get("uncertainty", ""), **values})
        return web.json_response({"songs": rows})
    except (ValueError, FileNotFoundError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)


async def review_caption(request):
    try:
        data = await request.json()
        path = caption_manifest(data["identifier"])
        result = read_json(path)
        song = next((s for s in result["songs"] if s["name"] == data["name"]), None)
        if song is None or not all(isinstance(data[k], str) and len(data[k]) <= 100000 for k in ("style", "lyrics")):
            raise ValueError("Invalid caption review")
        if not data["style"].strip():
            raise ValueError("Style caption cannot be empty")
        for key in ("style", "lyrics"):
            target = contained(result["directory"], song[key])
            write_text(target, data[key].strip())
        metadata_path = contained(result["directory"], song["metadata"])
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        metadata.update({"reviewed": True, "reviewed_text": fingerprint([data["style"].strip(), data["lyrics"].strip()]), "instrumental": not data["lyrics"].strip()})
        write_json(metadata_path, metadata)
        song.update(metadata)
        write_json(path, result)
        return web.json_response({"reviewed": True})
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        return web.json_response({"error": str(error)}, status=400)


async def get_run(request):
    try:
        name = run_name(request.match_info["name"])
        return web.json_response(read_run(output_root() / name / "run.json"))
    except (ValueError, FileNotFoundError) as error:
        return web.json_response({"error": str(error)}, status=400)


async def get_preview(request):
    try:
        name = run_name(request.match_info["name"])
        filename = request.match_info["filename"]
        root = output_root() / name
        run = read_run(root / "run.json")
        allowed = {c.get("preview") for c in run["checkpoints"]}
        allowed.add(run.get("baseline", {}).get("preview"))
        if filename not in allowed:
            raise ValueError("Preview is not part of this run")
        return web.FileResponse(contained(root, filename))
    except (ValueError, FileNotFoundError) as error:
        return web.json_response({"error": str(error)}, status=400)


if hasattr(PromptServer, "instance"):
    routes = PromptServer.instance.routes
    routes.get("/fl_yue2/captions/{identifier}")(get_captions)
    routes.post("/fl_yue2/captions/review")(review_caption)
    routes.get("/fl_yue2/training/run/{name}")(get_run)
    routes.get("/fl_yue2/training/audio/{name}/{filename}")(get_preview)
