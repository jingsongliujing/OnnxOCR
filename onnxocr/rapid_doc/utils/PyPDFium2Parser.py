import os
import tempfile
import threading


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _PdfiumProcessFileLock:
    def __init__(self):
        self._thread_lock = threading.RLock()
        self._local = threading.local()
        self._lock_path = os.getenv(
            "MINERU_PDFIUM_LOCK_FILE",
            os.path.join(tempfile.gettempdir(), "rapid_doc_pdfium.lock"),
        )
        self._process_lock_enabled = _env_flag("MINERU_PDFIUM_PROCESS_LOCK", False)

    def __enter__(self):
        self._thread_lock.acquire()

        depth = getattr(self._local, "depth", 0)
        if depth == 0 and self._process_lock_enabled:
            handle = open(self._lock_path, "a+b")
            self._acquire_file_lock(handle)
            self._local.handle = handle
        self._local.depth = depth + 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        depth = getattr(self._local, "depth", 0)
        if depth <= 1:
            handle = getattr(self._local, "handle", None)
            if handle is not None:
                try:
                    self._release_file_lock(handle)
                finally:
                    handle.close()
                    self._local.handle = None
            self._local.depth = 0
        else:
            self._local.depth = depth - 1

        self._thread_lock.release()
        return False

    @staticmethod
    def _acquire_file_lock(handle):
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _release_file_lock(handle):
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


lock = _PdfiumProcessFileLock()
