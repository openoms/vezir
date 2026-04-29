"""Multipart upload to the vezir service with retry."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import httpx

log = logging.getLogger("vezir.client.uploader")

ACCEPTED_AUDIO_EXTS = {".wav", ".ogg"}
CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024


def validate_audio_path(audio_path: Path) -> Path:
    """Validate a user-selected upload path and return it as a Path."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"audio path is not a file: {audio_path}")
    ext = audio_path.suffix.lower()
    if ext not in ACCEPTED_AUDIO_EXTS:
        allowed = ", ".join(sorted(ACCEPTED_AUDIO_EXTS))
        raise ValueError(f"unsupported audio type {ext or '(none)'}; expected {allowed}")
    return audio_path


def _progress_reporter(total_bytes: int):
    if not getattr(sys.stderr, "isatty", lambda: False)():
        return None

    last_percent = -1

    def report(sent_bytes: int) -> None:
        nonlocal last_percent
        percent = 100 if total_bytes == 0 else int((sent_bytes * 100) / total_bytes)
        if sent_bytes < total_bytes and percent == last_percent:
            return
        last_percent = percent
        total_mib = total_bytes / (1024 * 1024) if total_bytes else 0.0
        sent_mib = sent_bytes / (1024 * 1024)
        print(
            f"\rvezir: upload progress {percent:3d}% ({sent_mib:.1f}/{total_mib:.1f} MiB)",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if sent_bytes >= total_bytes:
            print(file=sys.stderr, flush=True)

    return report


def _start_chunked_upload(
    client: httpx.Client,
    server_url: str,
    headers: dict[str, str],
    audio_path: Path,
    content_type: str,
    title: str | None,
) -> dict:
    data = {
        "filename": audio_path.name,
        "size": str(audio_path.stat().st_size),
        "content_type": content_type,
    }
    if title:
        data["title"] = title
    resp = client.post(
        server_url.rstrip("/") + "/upload/start",
        headers=headers,
        data=data,
    )
    resp.raise_for_status()
    return resp.json()


def _upload_status(
    client: httpx.Client,
    server_url: str,
    headers: dict[str, str],
    session_id: str,
) -> int:
    resp = client.get(
        server_url.rstrip("/") + f"/upload/{session_id}/status",
        headers=headers,
    )
    resp.raise_for_status()
    return int(resp.json().get("received_bytes", 0))


def _complete_chunked_upload(
    client: httpx.Client,
    server_url: str,
    headers: dict[str, str],
    session_id: str,
) -> dict:
    resp = client.post(
        server_url.rstrip("/") + f"/upload/{session_id}/complete",
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def upload(
    server_url: str,
    token: str,
    audio_path: Path,
    title: str | None = None,
    timeout: float = 600.0,
    retries: int = 3,
) -> dict:
    """Upload audio to vezir using a chunked resumable protocol.

    Retries on connection errors and 5xx responses with exponential backoff.
    Raises httpx.HTTPError on permanent failure.
    """
    headers = {"Authorization": f"Bearer {token}"}

    audio_path = validate_audio_path(audio_path)

    ext = audio_path.suffix.lower()
    content_type = CONTENT_TYPES[ext]
    total_bytes = audio_path.stat().st_size
    report_progress = _progress_reporter(total_bytes)

    with httpx.Client(timeout=timeout) as client:
        state = _start_chunked_upload(client, server_url, headers, audio_path, content_type, title)
        session_id = state["session_id"]
        chunk_bytes = int(state.get("chunk_bytes", DEFAULT_CHUNK_BYTES))
        uploaded_bytes = int(state.get("received_bytes", 0))

        if report_progress:
            report_progress(uploaded_bytes)

        with audio_path.open("rb") as f:
            while uploaded_bytes < total_bytes:
                last_exc: Exception | None = None
                for attempt in range(1, retries + 1):
                    chunk_start = uploaded_bytes
                    f.seek(chunk_start)
                    chunk = f.read(min(chunk_bytes, total_bytes - chunk_start))
                    if not chunk:
                        uploaded_bytes = total_bytes
                        break

                    try:
                        resp = client.post(
                            server_url.rstrip("/") + f"/upload/{session_id}/chunk",
                            headers={
                                **headers,
                                "Content-Type": "application/octet-stream",
                                "X-Start-Byte": str(chunk_start),
                            },
                            content=chunk,
                        )
                        if resp.status_code == 200:
                            uploaded_bytes = int(resp.json()["received_bytes"])
                            if report_progress:
                                report_progress(uploaded_bytes)
                            break
                        if resp.status_code == 409:
                            resumed_bytes = _upload_status(client, server_url, headers, session_id)
                            if resumed_bytes > uploaded_bytes:
                                uploaded_bytes = resumed_bytes
                                if report_progress:
                                    report_progress(uploaded_bytes)
                            if resumed_bytes > chunk_start:
                                break
                        if 500 <= resp.status_code < 600:
                            log.warning(
                                "upload chunk attempt %d/%d: server %d %s",
                                attempt,
                                retries,
                                resp.status_code,
                                resp.text[:200],
                            )
                        resp.raise_for_status()
                    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                        log.warning(
                            "upload chunk attempt %d/%d failed: %s",
                            attempt,
                            retries,
                            exc,
                        )
                        last_exc = exc
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code < 500:
                            raise
                        last_exc = exc

                    if attempt < retries:
                        time.sleep(2 ** attempt)
                        resumed_bytes = _upload_status(client, server_url, headers, session_id)
                        if resumed_bytes > uploaded_bytes:
                            uploaded_bytes = resumed_bytes
                            if report_progress:
                                report_progress(uploaded_bytes)
                        if resumed_bytes > chunk_start:
                            break
                        if report_progress:
                            report_progress(uploaded_bytes)
                else:
                    if last_exc:
                        raise last_exc
                    raise RuntimeError(f"upload failed after {retries} attempts")

        return _complete_chunked_upload(client, server_url, headers, session_id)
