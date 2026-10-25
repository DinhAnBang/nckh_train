"""Create and validate reproducible bounded Network experiments.

The ``network_1m`` experiment samples each official split independently from
the already processed Parquet features.  It never reads the raw CSV files,
fits feature rules, or combines the official splits.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ton_iot.config.paths import FEATURES_DIR, NETWORK_1M_DATA_DIR
from .data_loader import EXPECTED_FEATURE_COUNT, inspect_layout
from .utils import sha256_file, write_json


SPLITS = ("train", "validation", "test")
TARGET_ROWS = {"train": 800_000, "validation": 100_000, "test": 100_000}
TIME_BIN_COUNT = 10
BATCH_SIZE = 50_000
SEED_OFFSETS = {"train": 0, "validation": 1_000_003, "test": 2_000_033}
FEATURE_KEY_DTYPE = np.dtype([("first", "<u8"), ("second", "<u8"), ("split", "u1")])


@dataclass(slots=True)
class ObservationBatch:
    relative: str
    row_start: int
    labels: np.ndarray
    types: list[str]
    timestamps: np.ndarray
    record_ids: list[str]
    source_files: list[str]
    source_rows: np.ndarray


@dataclass(slots=True)
class SampleRecord:
    relative: str
    row_in_part: int
    record_id: str
    timestamp: int
    label: int
    attack_type: str
    source_file: str
    source_row: int


def _normalise_type(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _split_parts(root: Path, split: str) -> list[Path]:
    parts = sorted((root / split / "X").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No X Parquet parts for {split}: {root / split / 'X'}")
    return parts


def _batch_column(batch: pa.RecordBatch, name: str):
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Column {name!r} is missing from sampling metadata")
    return batch.column(index)


def _iter_observation_batches(root: Path, split: str):
    """Yield aligned label/metadata batches without loading a full split."""

    metadata_columns = ["record_id", "source_file", "source_row", "ts", "type"]
    for x_path in _split_parts(root, split):
        relative = x_path.relative_to(root / split / "X").as_posix()
        y_path = root / split / "y" / relative
        metadata_path = root / split / "metadata" / relative
        y_file = pq.ParquetFile(y_path)
        metadata_file = pq.ParquetFile(metadata_path)
        if y_file.metadata.num_rows != metadata_file.metadata.num_rows:
            raise ValueError(f"Unaligned y/metadata rows: {split}/{relative}")
        y_batches = y_file.iter_batches(columns=["label"], batch_size=BATCH_SIZE)
        metadata_batches = metadata_file.iter_batches(columns=metadata_columns, batch_size=BATCH_SIZE)
        row_start = 0
        for y_batch, metadata_batch in itertools.zip_longest(y_batches, metadata_batches):
            if y_batch is None or metadata_batch is None or y_batch.num_rows != metadata_batch.num_rows:
                raise ValueError(f"Unaligned sampling batches: {split}/{relative}")
            labels = y_batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
            timestamps = _batch_column(metadata_batch, "ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            source_rows = _batch_column(metadata_batch, "source_row").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            types = [_normalise_type(value) for value in _batch_column(metadata_batch, "type").to_pylist()]
            record_ids = [str(value) for value in _batch_column(metadata_batch, "record_id").to_pylist()]
            source_files = [str(value) for value in _batch_column(metadata_batch, "source_file").to_pylist()]
            if any(not value for value in types) or any(value == "None" for value in record_ids):
                raise ValueError(f"Missing type or record_id in {split}/{relative}")
            yield ObservationBatch(
                relative=relative,
                row_start=row_start,
                labels=labels,
                types=types,
                timestamps=timestamps,
                record_ids=record_ids,
                source_files=source_files,
                source_rows=source_rows,
            )
            row_start += y_batch.num_rows


def _time_bins(timestamps: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    if maximum <= minimum:
        return np.zeros(len(timestamps), dtype=np.int16)
    width = maximum - minimum + 1
    bins = ((timestamps - minimum) * TIME_BIN_COUNT // width).astype(np.int16)
    return np.minimum(bins, TIME_BIN_COUNT - 1)


def _scan_split(root: Path, split: str) -> dict[str, object]:
    rows = 0
    labels = Counter()
    types = Counter()
    minimum = None
    maximum = None
    for batch in _iter_observation_batches(root, split):
        rows += len(batch.labels)
        labels.update(int(value) for value in batch.labels)
        types.update(batch.types)
        batch_minimum = int(batch.timestamps.min())
        batch_maximum = int(batch.timestamps.max())
        minimum = batch_minimum if minimum is None else min(minimum, batch_minimum)
        maximum = batch_maximum if maximum is None else max(maximum, batch_maximum)
    if rows == 0 or minimum is None or maximum is None:
        raise ValueError(f"Empty split: {split}")
    return {"rows": rows, "labels": labels, "types": types, "min_ts": minimum, "max_ts": maximum}


def _count_strata(root: Path, split: str, minimum: int, maximum: int) -> Counter:
    counts = Counter()
    for batch in _iter_observation_batches(root, split):
        bins = _time_bins(batch.timestamps, minimum, maximum)
        counts.update((kind, int(time_bin)) for kind, time_bin in zip(batch.types, bins))
    return counts


def _allocate_quotas(strata: Counter, target: int) -> dict[tuple[str, int], int]:
    available = sum(strata.values())
    if target > available:
        raise ValueError(f"Requested {target:,} rows but only {available:,} are available")
    raw = {key: strata[key] * target / available for key in strata}
    quotas = {key: min(strata[key], int(raw[key])) for key in strata}

    by_type: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(strata, key=str):
        by_type[key[0]].append(key)
    for keys in by_type.values():
        if sum(quotas[key] for key in keys) == 0:
            quotas[max(keys, key=lambda key: (strata[key], -key[1]))] = 1

    while sum(quotas.values()) < target:
        candidates = [key for key in strata if quotas[key] < strata[key]]
        if not candidates:
            raise ValueError("Could not allocate the requested sample size")
        key = max(candidates, key=lambda item: (raw[item] - quotas[item], strata[item] - quotas[item], str(item)))
        quotas[key] += 1

    while sum(quotas.values()) > target:
        candidates = [
            key for key in strata
            if quotas[key] > 0 and sum(quotas[item] for item in by_type[key[0]]) > 1
        ]
        if not candidates:
            raise ValueError("Minimum attack-type coverage exceeds the requested sample size")
        key = min(candidates, key=lambda item: (raw[item] - quotas[item], str(item)))
        quotas[key] -= 1
    return quotas


def _sample_split(root: Path, split: str, scan: dict[str, object], target: int, seed: int):
    strata = _count_strata(root, split, scan["min_ts"], scan["max_ts"])
    quotas = _allocate_quotas(strata, target)
    reservoirs: dict[tuple[str, int], list[SampleRecord]] = {key: [] for key in quotas}
    seen = Counter()
    rng = np.random.default_rng(seed + SEED_OFFSETS[split])
    for batch in _iter_observation_batches(root, split):
        bins = _time_bins(batch.timestamps, scan["min_ts"], scan["max_ts"])
        for index, (kind, time_bin) in enumerate(zip(batch.types, bins)):
            key = (kind, int(time_bin))
            quota = quotas.get(key, 0)
            if quota == 0:
                continue
            seen[key] += 1
            candidate = SampleRecord(
                relative=batch.relative,
                row_in_part=batch.row_start + index,
                record_id=batch.record_ids[index],
                timestamp=int(batch.timestamps[index]),
                label=int(batch.labels[index]),
                attack_type=kind,
                source_file=batch.source_files[index],
                source_row=int(batch.source_rows[index]),
            )
            reservoir = reservoirs[key]
            if len(reservoir) < quota:
                reservoir.append(candidate)
            else:
                replacement = int(rng.integers(0, seen[key]))
                if replacement < quota:
                    reservoir[replacement] = candidate
    selected = [item for reservoir in reservoirs.values() for item in reservoir]
    selected.sort(key=lambda item: (item.relative, item.row_in_part))
    if len(selected) != target:
        raise ValueError(f"{split} sample has {len(selected):,} rows; expected {target:,}")
    full_types = set(scan["types"])
    selected_types = {item.attack_type for item in selected}
    missing_types = sorted(full_types - selected_types)
    if missing_types:
        raise ValueError(f"{split} sample lost attack types: {missing_types}")
    if split == "test" and "mitm" in full_types and "mitm" not in selected_types:
        raise ValueError("Test sample lost MITM coverage")
    allocation = [
        {"type": key[0], "time_bin": key[1], "available": strata[key], "selected": quotas[key]}
        for key in sorted(strata, key=str)
    ]
    return selected, allocation


def _counts(records: list[SampleRecord], minimum: int, maximum: int) -> dict[str, object]:
    labels = Counter(item.label for item in records)
    types = Counter(item.attack_type for item in records)
    bins = Counter(int(_time_bins(np.array([item.timestamp]), minimum, maximum)[0]) for item in records)
    return {
        "rows": len(records),
        "labels": {str(key): value for key, value in sorted(labels.items())},
        "label_ratio": {str(key): round(value / len(records), 8) for key, value in sorted(labels.items())},
        "types": dict(sorted(types.items())),
        "time_bins": {str(key): value for key, value in sorted(bins.items())},
        "min_ts": min(item.timestamp for item in records),
        "max_ts": max(item.timestamp for item in records),
    }


def _selection_hash(records: list[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(f"{item.relative}\t{item.row_in_part}\t{item.record_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_sample_split(root: Path, staging: Path, split: str, records: list[SampleRecord]):
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.relative].append(record)
    writers = {}
    output_paths = {
        role: staging / split / role / "part-00000.parquet"
        for role in ("X", "y", "metadata")
    }
    try:
        for relative, selected in sorted(grouped.items()):
            selected.sort(key=lambda item: item.row_in_part)
            indexes = pa.array([item.row_in_part for item in selected], type=pa.int64())
            source_x = pq.read_table(root / split / "X" / relative).take(indexes)
            source_y = pq.read_table(root / split / "y" / relative).take(indexes)
            source_metadata = pq.read_table(root / split / "metadata" / relative).take(indexes)
            for role, table in (("X", source_x), ("y", source_y), ("metadata", source_metadata)):
                output_paths[role].parent.mkdir(parents=True, exist_ok=True)
                if role not in writers:
                    writers[role] = pq.ParquetWriter(output_paths[role], table.schema, compression="zstd")
                writers[role].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def _write_selection_records(staging: Path, selections: dict[str, list[SampleRecord]]):
    path = staging / "selection" / "selected_records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        rows = []
        for split in SPLITS:
            for item in selections[split]:
                rows.append({
                    "split": split,
                    "record_id": item.record_id,
                    "source_file": item.source_file,
                    "source_row": item.source_row,
                    "source_part": item.relative,
                    "row_in_part": item.row_in_part,
                    "ts": item.timestamp,
                    "label": item.label,
                    "type": item.attack_type,
                })
                if len(rows) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows = []
        if rows:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _feature_duplicate_summary(keys: list[np.ndarray]) -> dict[str, int]:
    if not keys:
        return {"groups": 0, "extra_rows": 0, "cross_split_groups": 0, "cross_split_rows": 0}
    values = np.concatenate(keys)
    order = np.lexsort((values["second"], values["first"]))
    values = values[order]
    different = (values["first"][1:] != values["first"][:-1]) | (values["second"][1:] != values["second"][:-1])
    starts = np.r_[0, np.flatnonzero(different) + 1]
    ends = np.r_[starts[1:], len(values)]
    groups = 0
    extras = 0
    cross_groups = 0
    cross_rows = 0
    for start, end in zip(starts, ends):
        if end - start <= 1:
            continue
        groups += 1
        extras += end - start - 1
        split_count = len(set(values["split"][start:end].tolist()))
        if split_count > 1:
            cross_groups += 1
            cross_rows += end - start
    return {
        "groups": int(groups),
        "extra_rows": int(extras),
        "cross_split_groups": int(cross_groups),
        "cross_split_rows": int(cross_rows),
    }


def validate_sample(root: Path, targets: dict[str, int] = TARGET_ROWS) -> dict[str, object]:
    """Validate the sample with bounded batches and independent feature hashes."""

    root = Path(root).expanduser().resolve()
    layout = inspect_layout(root, expected_rows=targets)
    split_stats = {}
    record_locations = {}
    feature_keys = []
    for split_index, split in enumerate(SPLITS):
        stats = {"rows": 0, "labels": Counter(), "types": Counter(), "min_ts": None, "max_ts": None}
        for x_path in layout["parts"][split]:
            _, y_path, metadata_path = (
                x_path,
                root / split / "y" / x_path.relative_to(root / split / "X"),
                root / split / "metadata" / x_path.relative_to(root / split / "X"),
            )
            x_file = pq.ParquetFile(x_path)
            y_file = pq.ParquetFile(y_path)
            metadata_file = pq.ParquetFile(metadata_path)
            x_batches = x_file.iter_batches(batch_size=BATCH_SIZE)
            y_batches = y_file.iter_batches(columns=["label"], batch_size=BATCH_SIZE)
            metadata_batches = metadata_file.iter_batches(columns=["record_id", "type", "ts"], batch_size=BATCH_SIZE)
            for x_batch, y_batch, metadata_batch in zip(x_batches, y_batches, metadata_batches):
                if not (x_batch.num_rows == y_batch.num_rows == metadata_batch.num_rows):
                    raise ValueError(f"X/y/metadata mismatch in sample: {split}")
                labels = y_batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError(f"Invalid binary label in sample: {split}")
                kinds = [_normalise_type(value) for value in _batch_column(metadata_batch, "type").to_pylist()]
                timestamps = _batch_column(metadata_batch, "ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                ids = [str(value) for value in _batch_column(metadata_batch, "record_id").to_pylist()]
                if any(not kind for kind in kinds) or any(value == "None" for value in ids):
                    raise ValueError(f"Missing metadata in sample: {split}")
                if not np.array_equal(np.array([kind == "normal" for kind in kinds]), labels == 0):
                    raise ValueError(f"Label/type mismatch in sample: {split}")
                for record_id in ids:
                    if record_id in record_locations:
                        raise ValueError(f"Duplicate record_id in sample: {record_id}")
                    record_locations[record_id] = split
                for index, field in enumerate(x_batch.schema):
                    values = x_batch.column(index).to_numpy(zero_copy_only=False)
                    if x_batch.column(index).null_count:
                        raise ValueError(f"Missing feature value in sample: {field.name}")
                    if pa.types.is_floating(field.type) and not np.isfinite(values).all():
                        raise ValueError(f"Non-finite feature value in sample: {field.name}")
                frame = x_batch.to_pandas()
                first = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
                second = pd.util.hash_pandas_object(frame.iloc[:, ::-1], index=False).to_numpy(dtype=np.uint64)
                batch_keys = np.empty(len(first), dtype=FEATURE_KEY_DTYPE)
                batch_keys["first"], batch_keys["second"], batch_keys["split"] = first, second, split_index
                feature_keys.append(batch_keys)
                stats["rows"] += x_batch.num_rows
                stats["labels"].update(int(value) for value in labels)
                stats["types"].update(kinds)
                minimum, maximum = int(timestamps.min()), int(timestamps.max())
                stats["min_ts"] = minimum if stats["min_ts"] is None else min(stats["min_ts"], minimum)
                stats["max_ts"] = maximum if stats["max_ts"] is None else max(stats["max_ts"], maximum)
        split_stats[split] = {
            "rows": stats["rows"],
            "labels": {str(key): value for key, value in sorted(stats["labels"].items())},
            "types": dict(sorted(stats["types"].items())),
            "min_ts": stats["min_ts"],
            "max_ts": stats["max_ts"],
        }
    if not (split_stats["train"]["max_ts"] < split_stats["validation"]["min_ts"]
            <= split_stats["validation"]["max_ts"] < split_stats["test"]["min_ts"]):
        raise ValueError("Sample time order across train/validation/test is not preserved")
    duplicate_summary = _feature_duplicate_summary(feature_keys)
    return {
        "status": "PASS",
        "rows": {split: split_stats[split]["rows"] for split in SPLITS},
        "features": len(layout["columns"]),
        "feature_columns": layout["columns"],
        "splits": split_stats,
        "same_feature_duplicates": duplicate_summary,
        "record_id_unique": True,
        "x_finite": True,
        "x_y_metadata_aligned": True,
        "time_order": True,
    }


def create_network_1m(source_dir=FEATURES_DIR, output_dir=NETWORK_1M_DATA_DIR, seed=20261025):
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Sample output already exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + "_staging")
    if staging.exists():
        raise FileExistsError(f"Incomplete sample staging directory exists: {staging}")
    inspect_layout(source_dir)
    rules_path = source_dir / "metadata" / "train_fitted_rules.json"
    staging.mkdir(parents=True)
    selections = {}
    scans = {}
    allocations = {}
    try:
        for split in SPLITS:
            scans[split] = _scan_split(source_dir, split)
            selections[split], allocations[split] = _sample_split(
                source_dir, split, scans[split], TARGET_ROWS[split], seed
            )
            _write_sample_split(source_dir, staging, split, selections[split])
        (staging / "metadata").mkdir(parents=True, exist_ok=True)
        shutil.copy2(rules_path, staging / "metadata" / "train_fitted_rules.json")
        _write_selection_records(staging, selections)
        validation = validate_sample(staging)
        report = {
            "experiment": "network_1m",
            "status": validation["status"],
            "seed": seed,
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "source_rules_sha256": sha256_file(rules_path),
            "feature_count": EXPECTED_FEATURE_COUNT,
            "targets": TARGET_ROWS,
            "input_rows": {split: scans[split]["rows"] for split in SPLITS},
            "input_labels": {split: dict(sorted(scans[split]["labels"].items())) for split in SPLITS},
            "input_types": {split: dict(sorted(scans[split]["types"].items())) for split in SPLITS},
            "input_time_range": {split: {"min_ts": scans[split]["min_ts"], "max_ts": scans[split]["max_ts"]} for split in SPLITS},
            "quotas": allocations,
            "sample": {
                split: {
                    **_counts(selections[split], scans[split]["min_ts"], scans[split]["max_ts"]),
                    "selection_sha256": _selection_hash(selections[split]),
                }
                for split in SPLITS
            },
            "validation": validation,
            "sampling_method": "Independent deterministic stratified reservoir sampling by attack type and 10 time bins; output kept in source-part order.",
            "raw_csv_read": False,
            "feature_rules_fitted_again": False,
        }
        write_json(staging / "sampling_report.json", report)
        staging.rename(output_dir)
    except Exception:
        # Preserve a failed staging directory for diagnosis; never replace an existing experiment.
        raise
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="Create the reproducible network_1m sample")
    sample.add_argument("--source-dir", type=Path, default=FEATURES_DIR)
    sample.add_argument("--output-dir", type=Path, default=NETWORK_1M_DATA_DIR)
    sample.add_argument("--seed", type=int, default=20261025)
    validate = subparsers.add_parser("validate", help="Validate an existing network_1m sample")
    validate.add_argument("--data-dir", type=Path, required=True)
    validate.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        report = create_network_1m(args.source_dir, args.output_dir, args.seed)
        print(json.dumps({"status": report["status"], "rows": report["sample"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
        return 0
    report = validate_sample(args.data_dir)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
