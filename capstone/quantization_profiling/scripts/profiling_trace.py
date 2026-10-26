"""Structured stage logs and optional NVTX, without importing torch or CUDA."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache, wraps
import importlib
import json
import logging
import os
from pathlib import Path
import threading
import time

DOMAIN = "quantization_profiling"
LOGGER = logging.getLogger("quantization_profiling.stages")
LOGGER.propagate = False


def configure_logging(path=None):
    close_logging()
    LOGGER.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[quant-profile] %(message)s"))
    LOGGER.addHandler(console)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = logging.FileHandler(path, encoding="utf-8")
        output.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(output)


def close_logging():
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()


def managed_logging(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        configure_logging()
        try:
            return function(*args, **kwargs)
        finally:
            close_logging()
    return wrapped


def log_event(event, **fields):
    if not LOGGER.isEnabledFor(logging.INFO) or not LOGGER.handlers:
        return
    record = {"utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(), "ppid": os.getppid(),
              "thread_id": threading.get_ident(), "native_thread_id": threading.get_native_id(),
              "event": event, **fields}
    message = json.dumps(record, ensure_ascii=False, default=str)
    secret = os.environ.get("VLLM_API_KEY", "")
    if secret:
        message = message.replace(secret, "<redacted>")
    LOGGER.info(message)


@lru_cache(maxsize=1)
def require_nvtx():
    if os.environ.get("NVTX_DISABLE", "0").strip().lower() not in ("", "0", "false"):
        raise RuntimeError("Unset NVTX_DISABLE before using --nvtx")
    try:
        return importlib.import_module("nvtx")
    except ImportError as error:
        raise RuntimeError("--nvtx requires the optional nvtx package; install requirements-profiling.txt") from error


@contextmanager
def nvtx_range(name, *, enabled=False, payload=None):
    if not enabled:
        yield
        return
    with require_nvtx().annotate(message=name, domain=DOMAIN, payload=payload):
        yield


class ProcessRange:
    """Explicitly ended range, safe for callbacks and overlapping requests."""
    def __init__(self, name, enabled=False):
        self.name, self.enabled = name, enabled
        self.handle = None
        self.active = False

    def start(self):
        if self.enabled and not self.active:
            self.handle = require_nvtx().start_range(message=self.name, domain=DOMAIN)
            self.active = True
        return self

    def finish(self):
        if self.active:
            require_nvtx().end_range(self.handle)
            self.active = False


@contextmanager
def stage(name, *, nvtx=False, emit_log=True, payload=None, **fields):
    started = time.perf_counter()
    if emit_log:
        log_event("stage_start", stage=name, **fields)
    try:
        with nvtx_range(name, enabled=nvtx, payload=payload):
            yield
    except BaseException as error:
        if emit_log:
            log_event("stage_end", stage=name, status="failed",
                      cpu_wall_seconds=time.perf_counter()-started,
                      error=str(error), **fields)
        raise
    else:
        if emit_log:
            log_event("stage_end", stage=name, status="ok",
                      cpu_wall_seconds=time.perf_counter()-started, **fields)
