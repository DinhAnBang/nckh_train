"""Run identity, resources, versions and small persistence utilities."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

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
