"""Move existing generated data aside, then verify every file by SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ton_iot.config.paths import BACKUP_PARENT as CONFIG_BACKUP_PARENT, PROJECT_ROOT


PROJECT = PROJECT_ROOT
BACKUP_PARENT = CONFIG_BACKUP_PARENT
SOURCES = (Path("dataPreprocessing/outputs"), Path("dataPreprocessing/reports"))
ACTIVE = ("cleaned", "split", "features")
GIB = 1024**3


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root)).replace("\\", "/")] = {
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.startswith("pipeline_") or not args.run_id.replace("_", "").isalnum():
        parser.error("Use a simple pipeline_YYYYMMDD_HHMMSS run ID")

    project = PROJECT.resolve(strict=True)
    parent = BACKUP_PARENT.resolve() if BACKUP_PARENT.exists() else BACKUP_PARENT.parent.resolve(strict=True) / BACKUP_PARENT.name
    destination = parent / args.run_id
    if parent == project or project in parent.parents or destination.exists():
        raise ValueError(f"Unsafe or existing backup destination: {destination}")
    source_roots = []
    for relative in SOURCES:
        source = (PROJECT / relative).resolve(strict=True)
        if project not in source.parents or source.is_symlink() or not source.is_dir():
            raise ValueError(f"Unsafe source: {source}")
        source_roots.append(source)
    if any(source.drive.lower() != destination.drive.lower() for source in source_roots):
        raise ValueError("Backup must stay on the same drive")

    before = {}
    for relative, source in zip(SOURCES, source_roots):
        before[str(relative).replace("\\", "/")] = inventory(source)
    old_bytes = sum(item["bytes"] for group in before.values() for item in group.values())
    active_bytes = sum(sum(path.stat().st_size for path in (source_roots[0] / name).rglob("*") if path.is_file())
                       for name in ACTIVE)
    free_bytes = shutil.disk_usage(project).free
    minimum_free = 20 * GIB + 4 * active_bytes
    print(json.dumps({"old_files": sum(map(len, before.values())), "old_bytes": old_bytes,
                      "active_output_bytes": active_bytes, "free_bytes": free_bytes,
                      "minimum_free_bytes": minimum_free, "backup": str(destination)}, indent=2), flush=True)
    if free_bytes < minimum_free:
        raise RuntimeError("Insufficient free space with safety reserve; nothing was moved")

    destination.mkdir(parents=True)
    for relative, source in zip(SOURCES, source_roots):
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()) or target.exists():
            raise ValueError(f"Unsafe target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        print(f"MOVED {source} -> {target}", flush=True)

    after = {str(relative).replace("\\", "/"): inventory(destination / relative)
             for relative in SOURCES}
    verified = before == after and all(not source.exists() for source in source_roots)
    manifest = {"run_id": args.run_id, "backup_path": str(destination),
                "old_files": sum(map(len, before.values())), "old_bytes": old_bytes,
                "active_output_bytes": active_bytes, "free_bytes_before": free_bytes,
                "minimum_free_bytes": minimum_free, "verified": verified,
                "files": after}
    (destination / "backup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not verified:
        raise RuntimeError("Backup verification failed; stop before running the pipeline")
    print(f"BACKUP VERIFIED: {destination}; files={manifest['old_files']}; bytes={old_bytes}", flush=True)


if __name__ == "__main__":
    main()
