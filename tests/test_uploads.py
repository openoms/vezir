from __future__ import annotations

import io
import stat
import tempfile
import wave
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def tmp_data(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("VEZIR_DATA", d)
        monkeypatch.delenv("VEZIR_MAX_UPLOAD_BYTES", raising=False)
        yield Path(d)


@pytest.fixture
def client_and_token(tmp_data):
    from fastapi.testclient import TestClient
    from vezir.server import auth
    from vezir.server.app import create_app

    token = auth.issue("alice")
    app = create_app()
    return TestClient(app, follow_redirects=False), token, tmp_data


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_upload_accepts_wav(client_and_token):
    client, token, tmp_data = client_and_token

    resp = client.post(
        "/upload",
        headers=_bearer(token),
        files={"audio": ("foo.wav", _wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 200
    body = resp.json()
    uploaded = tmp_data / "sessions" / body["session_id"] / f"{body['session_id']}.wav"
    assert uploaded.exists()
    assert _mode(uploaded.parent) == 0o700
    assert _mode(uploaded) == 0o600


def test_upload_accepts_ogg(client_and_token):
    client, token, tmp_data = client_and_token

    resp = client.post(
        "/upload",
        headers=_bearer(token),
        files={"audio": ("foo.ogg", b"OggS" + b"\x00" * 64, "audio/ogg")},
    )

    assert resp.status_code == 200
    body = resp.json()
    uploaded = tmp_data / "sessions" / body["session_id"] / f"{body['session_id']}.ogg"
    assert uploaded.exists()


def test_chunked_upload_accepts_wav(client_and_token):
    client, token, tmp_data = client_and_token
    wav_bytes = _wav_bytes()

    started = client.post(
        "/upload/start",
        headers=_bearer(token),
        data={
            "filename": "foo.wav",
            "size": str(len(wav_bytes)),
            "content_type": "audio/wav",
        },
    )

    assert started.status_code == 200
    session_id = started.json()["session_id"]

    chunked = client.post(
        f"/upload/{session_id}/chunk",
        headers={**_bearer(token), "X-Start-Byte": "0", "Content-Type": "application/octet-stream"},
        content=wav_bytes,
    )
    assert chunked.status_code == 200
    assert chunked.json()["received_bytes"] == len(wav_bytes)

    completed = client.post(f"/upload/{session_id}/complete", headers=_bearer(token))
    assert completed.status_code == 200
    body = completed.json()
    uploaded = tmp_data / "sessions" / body["session_id"] / f"{body['session_id']}.wav"
    assert uploaded.exists()
    assert uploaded.read_bytes() == wav_bytes


def test_chunked_upload_rejects_offset_mismatch(client_and_token):
    client, token, _tmp_data = client_and_token
    wav_bytes = _wav_bytes()
    first = wav_bytes[:64]

    started = client.post(
        "/upload/start",
        headers=_bearer(token),
        data={
            "filename": "foo.wav",
            "size": str(len(wav_bytes)),
            "content_type": "audio/wav",
        },
    )
    session_id = started.json()["session_id"]

    ok = client.post(
        f"/upload/{session_id}/chunk",
        headers={**_bearer(token), "X-Start-Byte": "0", "Content-Type": "application/octet-stream"},
        content=first,
    )
    assert ok.status_code == 200

    conflict = client.post(
        f"/upload/{session_id}/chunk",
        headers={**_bearer(token), "X-Start-Byte": "0", "Content-Type": "application/octet-stream"},
        content=wav_bytes[64:128],
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["received_bytes"] == len(first)

    status = client.get(f"/upload/{session_id}/status", headers=_bearer(token))
    assert status.status_code == 200
    assert status.json()["received_bytes"] == len(first)


def test_chunked_upload_complete_is_idempotent(client_and_token):
    client, token, tmp_data = client_and_token
    wav_bytes = _wav_bytes()

    started = client.post(
        "/upload/start",
        headers=_bearer(token),
        data={
            "filename": "foo.wav",
            "size": str(len(wav_bytes)),
            "content_type": "audio/wav",
        },
    )
    session_id = started.json()["session_id"]

    chunked = client.post(
        f"/upload/{session_id}/chunk",
        headers={**_bearer(token), "X-Start-Byte": "0", "Content-Type": "application/octet-stream"},
        content=wav_bytes,
    )
    assert chunked.status_code == 200

    first = client.post(f"/upload/{session_id}/complete", headers=_bearer(token))
    second = client.post(f"/upload/{session_id}/complete", headers=_bearer(token))

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["bytes"] == len(wav_bytes)
    uploaded = tmp_data / "sessions" / session_id / f"{session_id}.wav"
    assert uploaded.read_bytes() == wav_bytes


def test_upload_rejects_unknown_type(client_and_token):
    client, token, tmp_data = client_and_token

    resp = client.post(
        "/upload",
        headers=_bearer(token),
        files={"audio": ("foo.txt", b"not audio", "text/plain")},
    )

    assert resp.status_code == 415
    assert list((tmp_data / "sessions").iterdir()) == []


def test_upload_rejects_invalid_wav_header(client_and_token):
    client, token, tmp_data = client_and_token

    resp = client.post(
        "/upload",
        headers=_bearer(token),
        files={"audio": ("foo.wav", b"not a real wav", "audio/wav")},
    )

    assert resp.status_code == 415
    assert list((tmp_data / "sessions").iterdir()) == []


def test_upload_rejects_oversized_body(monkeypatch, client_and_token):
    client, token, tmp_data = client_and_token
    monkeypatch.setenv("VEZIR_MAX_UPLOAD_BYTES", "100")

    resp = client.post(
        "/upload",
        headers=_bearer(token),
        files={"audio": ("foo.wav", _wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 413
    assert list((tmp_data / "sessions").iterdir()) == []


def test_cli_upload_existing_file(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from vezir import cli
    from vezir.client import uploader

    audio = tmp_path / "prior.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    def fake_upload(server_url, token, audio_path, title=None):
        assert server_url == "http://server.test"
        assert token == "vzr_test"
        assert audio_path == audio
        assert title == "prior meeting"
        return {
            "session_id": "01TEST",
            "bytes": 12,
            "dashboard_url": "http://server.test/s/01TEST",
            "dashboard_login_url": "http://server.test/login?next=%2Fs%2F01TEST",
        }

    monkeypatch.setattr(uploader, "upload", fake_upload)
    result = CliRunner().invoke(
        cli.main,
        [
            "upload",
            str(audio),
            "--server",
            "http://server.test",
            "--token",
            "vzr_test",
            "--title",
            "prior meeting",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "uploaded as session 01TEST" in result.output


def test_client_upload_retries_chunk_and_resumes(monkeypatch, client_and_token, tmp_path):
    client, token, tmp_data = client_and_token
    from vezir.client import uploader

    audio = tmp_path / "prior.wav"
    wav_bytes = _wav_bytes()
    audio.write_bytes(wav_bytes)

    def _to_httpx_response(method: str, url: str, response) -> httpx.Response:
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=httpx.Request(method, url),
        )

    class FakeHttpxClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.failed_once = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, data=None, files=None, content=None):
            path = url.removeprefix("http://server.test")
            if path.endswith("/chunk") and not self.failed_once:
                self.failed_once = True
                return httpx.Response(
                    503,
                    text="temporary",
                    request=httpx.Request("POST", url),
                )
            response = client.post(path, headers=headers, data=data, files=files, content=content)
            return _to_httpx_response("POST", url, response)

        def get(self, url, headers=None):
            path = url.removeprefix("http://server.test")
            response = client.get(path, headers=headers)
            return _to_httpx_response("GET", url, response)

    monkeypatch.setattr(uploader.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(uploader.time, "sleep", lambda *_args, **_kwargs: None)

    result = uploader.upload("http://server.test", token, audio, title="prior meeting")

    assert result["session_id"]
    uploaded = tmp_data / "sessions" / result["session_id"] / f"{result['session_id']}.wav"
    assert uploaded.exists()
    assert uploaded.read_bytes() == wav_bytes


def test_client_upload_rereads_after_lost_chunk_response(monkeypatch, client_and_token, tmp_path):
    client, token, tmp_data = client_and_token
    from vezir.client import uploader

    audio = tmp_path / "prior.wav"
    wav_bytes = _wav_bytes()
    audio.write_bytes(wav_bytes)

    def _to_httpx_response(method: str, url: str, response, json_body=None) -> httpx.Response:
        if json_body is not None:
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                json=json_body,
                request=httpx.Request(method, url),
            )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=httpx.Request(method, url),
        )

    class FakeHttpxClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.lost_first_chunk_response = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, data=None, files=None, content=None):
            path = url.removeprefix("http://server.test")
            if path == "/upload/start":
                response = client.post(path, headers=headers, data=data, files=files, content=content)
                body = response.json()
                body["chunk_bytes"] = 64
                return _to_httpx_response("POST", url, response, json_body=body)
            if path.endswith("/chunk") and not self.lost_first_chunk_response:
                self.lost_first_chunk_response = True
                client.post(path, headers=headers, data=data, files=files, content=content)
                raise httpx.ReadTimeout("lost chunk response", request=httpx.Request("POST", url))
            response = client.post(path, headers=headers, data=data, files=files, content=content)
            return _to_httpx_response("POST", url, response)

        def get(self, url, headers=None):
            path = url.removeprefix("http://server.test")
            response = client.get(path, headers=headers)
            return _to_httpx_response("GET", url, response)

    monkeypatch.setattr(uploader.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(uploader.time, "sleep", lambda *_args, **_kwargs: None)

    result = uploader.upload("http://server.test", token, audio, title="prior meeting")

    uploaded = tmp_data / "sessions" / result["session_id"] / f"{result['session_id']}.wav"
    assert uploaded.read_bytes() == wav_bytes


def test_run_scribe_prefers_ogg_when_present(monkeypatch, tmp_path):
    from vezir.client import scribe

    session_dir = tmp_path / "meeting-123"
    session_dir.mkdir()
    wav = session_dir / "meeting.wav"
    ogg = session_dir / "meeting.ogg"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    ogg.write_bytes(b"OggS" + b"\x00" * 64)

    class DummyProc:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    captured = {}

    def fake_upload(server_url, token, audio_path, title=None):
        captured["server_url"] = server_url
        captured["token"] = token
        captured["audio_path"] = audio_path
        captured["title"] = title
        return {"session_id": "01TEST", "dashboard_url": "http://server.test/s/01TEST"}

    monkeypatch.setattr(scribe, "_meet_bin", lambda: "meet")
    monkeypatch.setattr(scribe.subprocess, "Popen", lambda cmd: DummyProc())
    monkeypatch.setattr(scribe, "_find_latest_session", lambda output_dir, before: session_dir)
    monkeypatch.setattr(scribe.uploader, "upload", fake_upload)

    scribe.run_scribe(
        server_url="http://server.test",
        token="vzr_test",
        title="standup",
        output_dir=tmp_path,
    )

    assert captured["audio_path"] == ogg
