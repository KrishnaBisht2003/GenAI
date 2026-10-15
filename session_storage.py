import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import MutableMapping
from pathlib import Path


SESSION_DATA_ROOT = Path(
    os.environ.get(
        "RAG_SESSION_DATA_DIR",
        Path(tempfile.gettempdir()) / "genai-rag-sessions",
    )
).expanduser()
SESSION_IDLE_TIMEOUT_SECONDS = int(
    os.environ.get("RAG_SESSION_IDLE_TIMEOUT_SECONDS", "1800")
)
CLEANUP_INTERVAL_SECONDS = 60
ACTIVITY_FILE = ".last_activity"

_cleanup_lock = threading.Lock()
_cleanup_worker_started = False


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def cleanup_expired_sessions(
    now: float | None = None,
    exclude_session_ids: set[str] | None = None,
) -> None:
    current_time = time.time() if now is None else now
    excluded = exclude_session_ids or set()
    SESSION_DATA_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    with _cleanup_lock:
        for session_dir in SESSION_DATA_ROOT.iterdir():
            if not session_dir.is_dir() or session_dir.name in excluded:
                continue
            try:
                uuid.UUID(session_dir.name)
            except ValueError:
                continue

            activity_file = session_dir / ACTIVITY_FILE
            try:
                last_active = activity_file.stat().st_mtime
            except FileNotFoundError:
                last_active = session_dir.stat().st_mtime

            if current_time - last_active > SESSION_IDLE_TIMEOUT_SECONDS:
                shutil.rmtree(session_dir, ignore_errors=True)


def _cleanup_worker() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        cleanup_expired_sessions()


def start_session_cleanup_service() -> None:
    global _cleanup_worker_started
    with _cleanup_lock:
        if _cleanup_worker_started:
            return
        worker = threading.Thread(
            target=_cleanup_worker,
            name="rag-session-cleanup",
            daemon=True,
        )
        worker.start()
        _cleanup_worker_started = True


def get_session_workspace(
    session_state: MutableMapping,
) -> tuple[str, Path, Path]:
    session_id = session_state.get("rag_session_id")
    if not isinstance(session_id, str):
        session_id = ""
    try:
        uuid.UUID(session_id)
    except ValueError:
        session_id = uuid.uuid4().hex
        session_state["rag_session_id"] = session_id

    _ensure_private_directory(SESSION_DATA_ROOT)
    session_dir = SESSION_DATA_ROOT / session_id
    upload_dir = session_dir / "uploads"
    chroma_dir = session_dir / "chroma"
    with _cleanup_lock:
        _ensure_private_directory(session_dir)
        _ensure_private_directory(upload_dir)
        _ensure_private_directory(chroma_dir)
        activity_file = session_dir / ACTIVITY_FILE
        activity_file.touch(mode=0o600, exist_ok=True)
        os.utime(activity_file, None)

    cleanup_expired_sessions(exclude_session_ids={session_id})
    start_session_cleanup_service()
    return session_id, upload_dir, chroma_dir