"""Search engine implementation using yt-dlp for fast video discovery."""

import logging

try:
    import yt_dlp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yt_dlp = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, wait_random  # type: ignore
except ImportError:  # pragma: no cover - optional dependency

    class _DummyWait:
        def __add__(self, other):
            return self

    def retry(*dargs, **dkwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return _DummyWait()

    def wait_random(*args, **kwargs):
        return _DummyWait()


logger = logging.getLogger(__name__)
