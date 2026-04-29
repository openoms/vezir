"""Upload endpoint.

POST /upload
    multipart/form-data with:
        audio: the .wav/.ogg file produced by `meet record` or `vezir upload`
        title: optional meeting title

    Returns: { "session_id": "<ulid>", "dashboard_url": "..." }

The uploaded WAV is stored at sessions/<id>/<id>.wav (single-channel or
dual-channel; meetscribe handles both). A new job is enqueued for the
worker to process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import threading

import ulid
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)

from .. import config
from . import auth, queue

log = logging.getLogger("vezir.uploads")

router = APIRouter()


CHUNK_BYTES = 4 * 1024 * 1024  # 4 MB
UPLOAD_META = ".upload.json"
PART_SUFFIX = ".part"
_LOCKS_GUARD = threading.Lock()
_UPLOAD_LOCKS: dict[str, asyncio.Lock] = {}

# Audio extensions vezir accepts. Meetscribe handles both WAV and OGG natively
# (see meet/cli.py:389-390 and meet/label.py:66-70).
ACCEPTED_EXTS = {".wav", ".ogg"}
CONTENT_TYPE_EXTS = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
}


def _pick_extension(upload_filename: str | None, content_type: str | None) -> str:
    """Choose the on-disk extension based on filename/MIME or reject."""
    if upload_filename:
        ext = Path(upload_filename).suffix.lower()
        if ext in ACCEPTED_EXTS:
            return ext
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct in CONTENT_TYPE_EXTS:
            return CONTENT_TYPE_EXTS[ct]
    allowed = ", ".join(sorted(ACCEPTED_EXTS))
    raise HTTPException(
        status_code=415,
        detail=f"unsupported audio type; expected {allowed}",
    )


def _validate_magic(ext: str, chunk: bytes) -> None:
    """Reject obvious filename/MIME spoofing for WAV and OGG uploads."""
    if not chunk:
        return
    ok = False
    if ext == ".wav":
        ok = len(chunk) >= 12 and chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE"
    elif ext == ".ogg":
        ok = chunk.startswith(b"OggS")
    if not ok:
        raise HTTPException(status_code=415, detail=f"invalid {ext} audio header")


def _session_lock(session_id: str) -> asyncio.Lock:
    with _LOCKS_GUARD:
        lock = _UPLOAD_LOCKS.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _UPLOAD_LOCKS[session_id] = lock
        return lock


def _session_dir(session_id: str) -> Path:
    return config.sessions_dir() / session_id


def _meta_path(session_id: str) -> Path:
    return _session_dir(session_id) / UPLOAD_META


def _load_meta(session_id: str) -> dict:
    path = _meta_path(session_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="upload session not found") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="upload session metadata is invalid") from exc


def _write_meta(session_id: str, payload: dict) -> None:
    config.secure_write_text(_meta_path(session_id), json.dumps(payload, sort_keys=True))


def _part_path(session_id: str, ext: str) -> Path:
    return _session_dir(session_id) / f"{session_id}{ext}{PART_SUFFIX}"


def _final_path(session_id: str, ext: str) -> Path:
    return _session_dir(session_id) / f"{session_id}{ext}"


def _upload_response(request: Request, session_id: str, bytes_written: int) -> dict:
    base = str(request.base_url).rstrip("/")
    from urllib.parse import quote

    auth_token = request.headers.get("authorization", "")
    if auth_token.lower().startswith("bearer "):
        plaintext = auth_token.split(None, 1)[1].strip()
        login_url = (
            f"{base}/login?token={quote(plaintext, safe='')}"
            f"&next=%2Fs%2F{session_id}"
        )
    else:
        login_url = f"{base}/login?next=%2Fs%2F{session_id}"
    return {
        "session_id": session_id,
        "bytes": bytes_written,
        "dashboard_url": f"{base}/s/{session_id}",
        "dashboard_login_url": login_url,
    }


def _assert_upload_owner(meta: dict, github: str) -> None:
    if meta.get("github") != github:
        raise HTTPException(status_code=403, detail="upload belongs to another token")


def _received_bytes(session_id: str, ext: str, completed: bool) -> int:
    path = _final_path(session_id, ext) if completed else _part_path(session_id, ext)
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _enqueue_completed_upload(meta: dict, session_id: str) -> None:
    queue.enqueue(session_id, github=meta["github"], title=meta.get("title"))


async def _read_bounded_chunk(
    request: Request,
    current: int,
    expected: int | None,
) -> bytes:
    max_bytes = config.max_upload_bytes()
    body = bytearray()
    async for piece in request.stream():
        if not piece:
            continue
        new_size = current + len(body) + len(piece)
        if len(body) + len(piece) > CHUNK_BYTES:
            raise HTTPException(status_code=413, detail="chunk too large")
        if new_size > max_bytes:
            raise HTTPException(status_code=413, detail="upload too large")
        if expected is not None and new_size > expected:
            raise HTTPException(status_code=409, detail={"received_bytes": current})
        body.extend(piece)
    if not body:
        raise HTTPException(status_code=400, detail="empty chunk")
    return bytes(body)


@router.post("/upload/start")
async def start_upload(
    filename: str = Form(...),
    size: int = Form(...),
    title: str | None = Form(default=None),
    content_type: str | None = Form(default=None),
    github: str = Depends(auth.require_bearer),
):
    config.ensure_dirs()
    max_bytes = config.max_upload_bytes()
    if size < 0:
        raise HTTPException(status_code=400, detail="size must be non-negative")
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="upload too large")

    ext = _pick_extension(filename, content_type)
    session_id = ulid.new().str
    sdir = _session_dir(session_id)
    config.secure_mkdir(sdir)
    part = _part_path(session_id, ext)
    part.touch()
    config.secure_chmod_file(part)
    _write_meta(
        session_id,
        {
            "completed": False,
            "content_type": content_type,
            "expected_bytes": size,
            "ext": ext,
            "filename": filename,
            "github": github,
            "title": title,
        },
    )
    return {
        "session_id": session_id,
        "chunk_bytes": CHUNK_BYTES,
        "received_bytes": 0,
    }


@router.get("/upload/{session_id}/status")
async def upload_status(
    session_id: str,
    github: str = Depends(auth.require_bearer),
):
    meta = _load_meta(session_id)
    _assert_upload_owner(meta, github)
    received = _received_bytes(session_id, meta["ext"], bool(meta.get("completed")))
    return {
        "session_id": session_id,
        "chunk_bytes": CHUNK_BYTES,
        "completed": bool(meta.get("completed")),
        "expected_bytes": meta.get("expected_bytes"),
        "received_bytes": received,
    }


@router.post("/upload/{session_id}/chunk")
async def upload_chunk(
    session_id: str,
    request: Request,
    x_start_byte: int = Header(..., alias="X-Start-Byte"),
    github: str = Depends(auth.require_bearer),
):
    async with _session_lock(session_id):
        meta = _load_meta(session_id)
        _assert_upload_owner(meta, github)
        if meta.get("completed"):
            received = _received_bytes(session_id, meta["ext"], completed=True)
            raise HTTPException(status_code=409, detail={"received_bytes": received})

        ext = meta["ext"]
        current = _received_bytes(session_id, ext, completed=False)
        if x_start_byte != current:
            raise HTTPException(status_code=409, detail={"received_bytes": current})

        expected = meta.get("expected_bytes")
        chunk = await _read_bounded_chunk(request, current, expected)

        if current == 0:
            _validate_magic(ext, chunk)

        new_size = current + len(chunk)
        part = _part_path(session_id, ext)
        with part.open("ab") as f:
            f.write(chunk)
        config.secure_chmod_file(part)
        return {
            "session_id": session_id,
            "received_bytes": new_size,
        }


@router.post("/upload/{session_id}/complete")
async def complete_upload(
    session_id: str,
    request: Request,
    github: str = Depends(auth.require_bearer),
):
    async with _session_lock(session_id):
        meta = _load_meta(session_id)
        _assert_upload_owner(meta, github)
        ext = meta["ext"]
        if meta.get("completed"):
            current = _received_bytes(session_id, ext, completed=True)
            return _upload_response(request, session_id, current)

        part = _part_path(session_id, ext)
        current = _received_bytes(session_id, ext, completed=False)
        expected = meta.get("expected_bytes")
        if expected is not None and current != expected:
            raise HTTPException(
                status_code=409,
                detail={"received_bytes": current, "expected_bytes": expected},
            )

        final_path = _final_path(session_id, ext)
        part.replace(final_path)
        config.secure_chmod_file(final_path)
        meta["completed"] = True
        _write_meta(session_id, meta)

        log.info(
            "chunked upload accepted: session=%s github=%s bytes=%d ext=%s title=%r",
            session_id,
            meta["github"],
            current,
            ext,
            meta.get("title"),
        )
        _enqueue_completed_upload(meta, session_id)
        return _upload_response(request, session_id, current)


@router.post("/upload")
async def upload(
    request: Request,
    audio: UploadFile = File(...),
    title: str | None = Form(default=None),
    github: str = Depends(auth.require_bearer),
):
    config.ensure_dirs()
    max_bytes = config.max_upload_bytes()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=413, detail="upload too large")
        except ValueError:
            pass

    ext = _pick_extension(audio.filename, audio.content_type)
    session_id = ulid.new().str
    sdir = config.sessions_dir() / session_id
    config.secure_mkdir(sdir)
    out = sdir / f"{session_id}{ext}"

    bytes_written = 0
    try:
        with out.open("wb") as f:
            config.secure_chmod_file(out)
            first_chunk = True
            while True:
                chunk = await audio.read(CHUNK_BYTES)
                if not chunk:
                    break
                if first_chunk:
                    _validate_magic(ext, chunk)
                    first_chunk = False
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(status_code=413, detail="upload too large")
                f.write(chunk)
        config.secure_chmod_file(out)
    except HTTPException:
        out.unlink(missing_ok=True)
        try:
            sdir.rmdir()
        except OSError:
            pass
        raise

    log.info(
        "upload accepted: session=%s github=%s bytes=%d ext=%s title=%r",
        session_id, github, bytes_written, ext, title,
    )

    queue.enqueue(session_id, github=github, title=title)
    return _upload_response(request, session_id, bytes_written)
