import asyncio
from types import SimpleNamespace

from fl_yue2.yue2.training import routes
from fl_yue2.yue2.training.data import write_json


def test_caption_audio_is_manifest_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, 'output_root', lambda: tmp_path)
    identifier = 'a' * 24
    source = tmp_path / 'source'
    source.mkdir()
    audio = source / 'track.wav'
    audio.write_bytes(b'audio')
    manifest = routes.caption_manifest(identifier)
    data = {'directory': str(manifest.parent), 'source_directory': str(source), 'songs': [{'name': 'track', 'audio': str(audio)}]}
    write_json(manifest, data)
    request = SimpleNamespace(match_info={'identifier': identifier, 'name': 'track'})
    assert asyncio.run(routes.get_caption_audio(request)).status == 200
    request.match_info['name'] = '../track'
    assert asyncio.run(routes.get_caption_audio(request)).status == 400
    request.match_info['name'] = 'track'
    data['songs'][0]['audio'] = str(tmp_path / 'outside.wav')
    write_json(manifest, data)
    assert asyncio.run(routes.get_caption_audio(request)).status == 400
    request.match_info['identifier'] = '../outside'
    assert asyncio.run(routes.get_caption_audio(request)).status == 400
