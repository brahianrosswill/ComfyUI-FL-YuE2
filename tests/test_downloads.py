import hashlib
import io
import json

import pytest
from fl_yue2 import downloads


def installation(root):
    target = root / "YuE2-Vae"
    target.mkdir()
    for name in downloads.COMMON_FILES:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    weight = b"test checkpoint"
    (target / "model.safetensors").write_bytes(weight)
    (target / "weights_manifest.json").write_text(json.dumps({"files": {"model.safetensors": {
        "bytes": len(weight), "sha256": hashlib.sha256(weight).hexdigest()}}}))
    return target


def test_offline_verification_and_changed_weights(monkeypatch, tmp_path):
    target = installation(tmp_path)
    monkeypatch.setattr(downloads.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    monkeypatch.setattr(downloads, "urlopen", lambda *a, **k: pytest.fail("Offline loading contacted the network"))
    assert downloads.resolve("YuE2-Vae", False) == target
    assert (target / ".verified.json").is_file()
    assert downloads.resolve("YuE2-Vae", False) == target
    (target / "model.safetensors").write_bytes(b"bad checkpoint")
    with pytest.raises(ValueError, match="Corrupt"):
        downloads.resolve("YuE2-Vae", False)


@pytest.mark.parametrize("supports_range", [False, True])
def test_resume_and_server_ignoring_range(monkeypatch, tmp_path, supports_range):
    target = tmp_path / "weights"
    partial = tmp_path / "weights.partial"
    partial.write_bytes(b"abc")
    body = b"def" if supports_range else b"abcdef"
    response = io.BytesIO(body)
    response.status = 206 if supports_range else 200
    response.headers = {"Content-Length": str(len(body)), "Content-Range": "bytes 3-5/6"}
    def open_request(request, **kwargs):
        assert request.get_header("Range") == "bytes=3-"
        return response
    monkeypatch.setattr(downloads, "urlopen", open_request)
    downloads.transfer("YuE2-Vae", "revision", "model.safetensors", target)
    assert target.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_failed_transfer_keeps_partial(monkeypatch, tmp_path):
    target = tmp_path / "weights"
    response = io.BytesIO(b"abc")
    response.status = 200
    response.headers = {"Content-Length": "6"}
    monkeypatch.setattr(downloads, "urlopen", lambda *a, **k: response)
    with pytest.raises(IOError, match="Incomplete"):
        downloads.transfer("YuE2-Vae", "revision", "model.safetensors", target)
    assert not target.exists()
    assert target.with_name("weights.partial").read_bytes() == b"abc"


def test_complete_external_install_wins_over_partial(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "YuE2-Vae").mkdir(parents=True)
    (first / "YuE2-Vae" / "model.safetensors").write_bytes(b"partial")
    second.mkdir()
    complete = installation(second)
    monkeypatch.setattr(downloads.folder_paths, "get_folder_paths", lambda _: [str(first), str(second)])
    assert downloads.resolve("YuE2-Vae", False) == complete
