"""Run identity, resources, versions and small persistence utilities."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import threading
from time import monotonic

import psutil


GIB = 1024**3


def run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_resources(path):
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(path)
    return {"ram_total_gib": round(memory.total / GIB, 2),
            "ram_available_gib": round(memory.available / GIB, 2),
            "disk_free_gib": round(disk.free / GIB, 2),
            "cpu_count": psutil.cpu_count(logical=True)}


def gpu_status():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "nvidia-smi unavailable; CUDA not verified"
    result = subprocess.run([executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def gpu_memory_gib():
    """Return current GPU used/total memory, or ``None`` when unavailable."""

    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run(
        [executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        used, total = (float(value.strip()) / 1024 for value in result.stdout.splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"used_gib": round(used, 3), "total_gib": round(total, 3)}


class ResourceMonitor:
    """Sample process RAM and GPU memory while a model is fitting."""

    def __init__(self, interval_seconds=1.0):
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss_gib = rss_gib()
        current_gpu = gpu_memory_gib()
        self.peak_gpu_used_gib = current_gpu["used_gib"] if current_gpu else None
        self.started_at = None
        self.elapsed_seconds = 0.0

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss_gib = max(self.peak_rss_gib, rss_gib())
            current_gpu = gpu_memory_gib()
            if current_gpu:
                self.peak_gpu_used_gib = max(self.peak_gpu_used_gib or 0.0, current_gpu["used_gib"])
            self._stop.wait(self.interval_seconds)

    def start(self):
        self.started_at = monotonic()
        self._thread = threading.Thread(target=self._sample, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        if self.started_at is not None:
            self.elapsed_seconds = round(monotonic() - self.started_at, 3)
        self.peak_rss_gib = max(self.peak_rss_gib, rss_gib())
        return {
            "peak_rss_gib": round(self.peak_rss_gib, 3),
            "peak_gpu_used_gib": None if self.peak_gpu_used_gib is None else round(self.peak_gpu_used_gib, 3),
            "monitor_seconds": self.elapsed_seconds,
        }


def library_versions():
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def rss_gib():
    return round(psutil.Process().memory_info().rss / GIB, 3)
