"""Independent, part-wise audit of the newly produced Network features."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ton_iot.config.paths import DATA_OUTPUT_DIR, DATA_PREPROCESSING_DIR, PROJECT_ROOT, REPORT_OUTPUT_DIR


PROJECT = PROJECT_ROOT
DATA = DATA_PREPROCESSING_DIR
FEATURES = DATA_OUTPUT_DIR / "features"
SPLIT = DATA_OUTPUT_DIR / "split"
SPLITS = ("train", "validation", "test")
MISSING = {"", "null", "none", "nan", "na", "n/a"}
META_COLUMNS = ("record_id", "source_file", "source_row", "duplicate_key", "ts",
                "src_ip", "dst_ip", "src_port", "dst_port", "proto", "service", "type")
KEY_DTYPE = np.dtype([("first", "<u8"), ("second", "<u8"), ("split", "u1")])
EXPECTED_ATTACKS = {
    "train": {"scanning", "dos", "injection", "ddos"},
    "validation": {"password"},
    "test": {"xss", "ransomware", "backdoor", "mitm"},
}
GIB = 1024**3


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def files_under(root):
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def append_keys(handle, first, second, split_index):
    keys = np.empty(len(first), dtype=KEY_DTYPE)
    keys["first"], keys["second"], keys["split"] = first, second, split_index
    keys.tofile(handle)


def duplicate_counts(path, count):
    """Count repeated two-hash keys using a disk-backed array."""
    if path.stat().st_size != count * KEY_DTYPE.itemsize:
        raise ValueError(f"Incomplete duplicate key file: {path}")
    keys = np.memmap(path, dtype=KEY_DTYPE, mode="r+", shape=(count,))
    keys.sort(order=["first", "second"])
    result = {"groups": 0, "extra_rows": 0, "cross_split_groups": 0,
              "cross_split_rows": 0, "within_split_extra": {name: 0 for name in SPLITS}}
    start = 0
    while start < count:
        stop = min(start + 500_000, count)
        while stop < count and keys[stop - 1]["first"] == keys[stop]["first"] and keys[stop - 1]["second"] == keys[stop]["second"]:
            stop += 1
        block = keys[start:stop]
        distinct = (block["first"][1:] != block["first"][:-1]) | (block["second"][1:] != block["second"][:-1])
        left = np.r_[0, np.flatnonzero(distinct) + 1]
        right = np.r_[left[1:], len(block)]
        lengths = right - left
        repeated = lengths > 1
        result["groups"] += int(repeated.sum())
        result["extra_rows"] += int((lengths[repeated] - 1).sum())
        if repeated.any():
            minimum = np.minimum.reduceat(block["split"], left)
            maximum = np.maximum.reduceat(block["split"], left)
            cross = repeated & (minimum != maximum)
            result["cross_split_groups"] += int(cross.sum())
            result["cross_split_rows"] += int(lengths[cross].sum())
            for index, name in enumerate(SPLITS):
                per_group = np.add.reduceat((block["split"] == index).astype(np.int32), left)
                result["within_split_extra"][name] += int(np.maximum(per_group - 1, 0).sum())
        start = stop
    keys._mmap.close()
    return result


def check_feature_values(source, x, rules, train_fit, issues):
    """Recalculate every selected output cell from the saved split and rules."""
    selected = list(dict.fromkeys(rules["numeric"] + rules["categorical"]))
    frame = source.select(selected).to_pandas()
    for name in rules["numeric"]:
        values = frame[name]
        if train_fit is not None:
            minimum = values.min(skipna=True)
            if pd.notna(minimum):
                old = train_fit["minimum"].get(name)
                number = minimum.item() if hasattr(minimum, "item") else minimum
                train_fit["minimum"][name] = number if old is None else min(old, number)
        field = x.schema.field(name)
        dtype = np.int64 if pa.types.is_integer(field.type) else np.float64
        expected = values.fillna(rules["numeric_fill"][name]).to_numpy(dtype=dtype)
        actual = x.column(name).to_numpy(zero_copy_only=False)
        if not np.array_equal(actual, expected):
            issues["feature_cell_mismatch"] += int(np.count_nonzero(actual != expected))
        flag = name + "__was_missing"
        if flag in rules["feature_columns"]:
            actual_flag = x.column(flag).to_numpy(zero_copy_only=False)
            issues["feature_cell_mismatch"] += int(np.count_nonzero(actual_flag != values.isna().to_numpy()))
    for name in rules["categorical"]:
        values = frame[name].astype("string")
        absent = (values.isna() | values.str.strip().str.lower().isin(MISSING)).fillna(False)
        vocabulary = rules["categorical_vocabulary"][name]
        if train_fit is not None:
            train_fit["categories"][name].update(values[~absent].dropna().tolist())
        known = values.isin(vocabulary).fillna(False)
        for index, category in enumerate(vocabulary):
            column = f"{name}__cat_{index:03}"
            expected = values.eq(category).fillna(False).to_numpy(dtype=bool)
            actual = x.column(column).to_numpy(zero_copy_only=False)
            issues["feature_cell_mismatch"] += int(np.count_nonzero(actual != expected))
        for column, expected in ((name + "__missing", absent.to_numpy(dtype=bool)),
                                 (name + "__unknown", ((~absent) & (~known)).to_numpy(dtype=bool))):
            actual = x.column(column).to_numpy(zero_copy_only=False)
            issues["feature_cell_mismatch"] += int(np.count_nonzero(actual != expected))


def audit_part(split, index, relative, rules, stats, fitted, issues, handles, file_rows, feature_stats):
    paths = {
        "source": SPLIT / split / relative,
        "X": FEATURES / split / "X" / relative,
        "y": FEATURES / split / "y" / relative,
        "metadata": FEATURES / split / "metadata" / relative,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Missing paired part: {split}/{relative}")
    for role, path in paths.items():
        file_rows.append({"split": split, "role": role, "path": str(path.relative_to(PROJECT)).replace("\\", "/"),
                          "bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows})
        stats[split]["bytes"][role] += path.stat().st_size
    selected = list(dict.fromkeys(rules["numeric"] + rules["categorical"] + ["label"] + list(META_COLUMNS)))
    source = pq.read_table(paths["source"], columns=selected)
    x, y, metadata = (pq.read_table(paths[role]) for role in ("X", "y", "metadata"))
    rows = source.num_rows
    if any(table.num_rows != rows for table in (x, y, metadata)):
        raise ValueError(f"X/y/metadata row mismatch: {split}/{relative}")
    if x.schema.names != rules["feature_columns"]:
        raise ValueError(f"Feature name/order mismatch: {split}/{relative}")
    dtypes = [(field.name, str(field.type)) for field in x.schema]
    if feature_stats["schema"] is None:
        feature_stats["schema"] = dtypes
    elif dtypes != feature_stats["schema"]:
        raise ValueError(f"Feature dtype mismatch: {split}/{relative}")
    if y.schema.names != ["label"] or set(metadata.schema.names) != set(META_COLUMNS):
        raise ValueError(f"Unexpected label or metadata columns: {split}/{relative}")
    for name in META_COLUMNS:
        if not source.column(name).equals(metadata.column(name)):
            issues["metadata_cell_mismatch"] += 1
    actual_y = y.column("label").to_numpy(zero_copy_only=False)
    expected_y = source.column("label").to_pandas().astype("uint8").to_numpy()
    issues["label_cell_mismatch"] += int(np.count_nonzero(actual_y != expected_y))
    types = metadata.column("type").to_pandas().astype("string").str.strip().str.lower()
    issues["label_type_mismatch"] += int(np.count_nonzero((types == "normal").to_numpy(dtype=bool) != (actual_y == 0)))
    issues["invalid_labels"] += int(np.count_nonzero(~np.isin(actual_y, [0, 1])))
    stats[split]["labels"][0] += int(np.count_nonzero(actual_y == 0))
    stats[split]["labels"][1] += int(np.count_nonzero(actual_y == 1))
    stats[split]["types"].update(types.value_counts().to_dict())
    times = metadata.column("ts").to_numpy(zero_copy_only=False)
    stats[split]["min_ts"] = min(stats[split]["min_ts"], int(times.min()))
    stats[split]["max_ts"] = max(stats[split]["max_ts"], int(times.max()))
    stats[split]["rows"] += rows
    stats[split]["parts"] += 1
    check_feature_values(source, x, rules, fitted if split == "train" else None, issues)
    for name in rules["feature_columns"]:
        column = x.column(name)
        item = feature_stats["values"].setdefault(name, {"dtype": str(column.type), "nulls": 0,
                                                          "nonfinite": 0, "min": None, "max": None})
        item["nulls"] += column.null_count
        if column.null_count:
            issues["feature_nulls"] += column.null_count
        array = column.to_numpy(zero_copy_only=False)
        finite = np.isfinite(array) if not column.null_count else np.isfinite(pd.to_numeric(column.to_pandas(), errors="coerce"))
        bad = int(np.count_nonzero(~finite)) - column.null_count
        item["nonfinite"] += max(bad, 0)
        issues["feature_nonfinite"] += max(bad, 0)
        if finite.any():
            minimum, maximum = array[finite].min(), array[finite].max()
            item["min"] = float(minimum) if item["min"] is None else min(item["min"], float(minimum))
            item["max"] = float(maximum) if item["max"] is None else max(item["max"], float(maximum))
            stats[split]["nnz"] += int(np.count_nonzero(array[finite]))
        if pa.types.is_integer(column.type) and len(array) and (array.min() < -(2**53) or array.max() > 2**53):
            issues["integer_precision_outside_float64"] += int(np.count_nonzero((array < -(2**53)) | (array > 2**53)))
    x_frame = x.to_pandas()
    first = pd.util.hash_pandas_object(x_frame, index=False).to_numpy(dtype=np.uint64)
    second = pd.util.hash_pandas_object(x_frame.iloc[:, ::-1], index=False).to_numpy(dtype=np.uint64)
    append_keys(handles["features"], first, second, index)
    raw_keys = metadata.column("duplicate_key").to_pylist()
    if any(key is None or len(key) != 32 for key in raw_keys):
        raise ValueError(f"Invalid source duplicate key: {split}/{relative}")
    first = np.fromiter((int(key[:16], 16) for key in raw_keys), dtype=np.uint64, count=rows)
    second = np.fromiter((int(key[16:], 16) for key in raw_keys), dtype=np.uint64, count=rows)
    append_keys(handles["source"], first, second, index)
    ids = metadata.column("record_id").to_pandas()
    first = pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype=np.uint64)
    second = pd.util.hash_pandas_object(ids, index=False, hash_key="0123456789abcdef").to_numpy(dtype=np.uint64)
    append_keys(handles["record_id"], first, second, index)


def run_audit(run_id):
    report = REPORT_OUTPUT_DIR / run_id
    log_path = report / "execution_log.txt"
    if not log_path.is_file() or "PIPELINE PASS" not in log_path.read_text(encoding="utf-8"):
        raise ValueError("Full pipeline dry-run must pass before the final audit")
    rules = json.loads((FEATURES / "metadata" / "train_fitted_rules.json").read_text(encoding="utf-8"))
    stats = {name: {"rows": 0, "parts": 0, "labels": Counter(), "types": Counter(),
                    "bytes": Counter(), "min_ts": 2**63, "max_ts": -(2**63), "nnz": 0} for name in SPLITS}
    fitted = {"minimum": {}, "categories": {name: set() for name in rules["categorical"]}}
    issues = Counter()
    feature_stats = {"schema": None, "values": {}}
    file_rows = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"{now()} BEGIN independent_final_audit\n")
        with tempfile.TemporaryDirectory(prefix="pipeline_audit_", dir=report) as temporary:
            temp = Path(temporary)
            paths = {name: temp / (name + ".bin") for name in ("source", "features", "record_id")}
            handles = {name: path.open("wb") for name, path in paths.items()}
            try:
                for index, split in enumerate(SPLITS):
                    source_parts = files_under(SPLIT / split)
                    x_parts = files_under(FEATURES / split / "X")
                    y_parts = files_under(FEATURES / split / "y")
                    meta_parts = files_under(FEATURES / split / "metadata")
                    source_rel = {path.relative_to(SPLIT / split) for path in source_parts}
                    if any({path.relative_to(FEATURES / split / branch) for path in parts} != source_rel
                           for branch, parts in (("X", x_parts), ("y", y_parts), ("metadata", meta_parts))):
                        raise ValueError(f"Part filenames differ across source/X/y/metadata: {split}")
                    for part_number, relative in enumerate(sorted(source_rel), start=1):
                        audit_part(split, index, relative, rules, stats, fitted, issues, handles, file_rows, feature_stats)
                        if part_number % 25 == 0:
                            message = f"{now()} audited {split} {part_number}/{len(source_rel)} parts"
                            print(message, flush=True)
                            log.write(message + "\n")
                for handle in handles.values():
                    handle.close()
                total_rows = sum(item["rows"] for item in stats.values())
                duplicates = {name: duplicate_counts(path, total_rows) for name, path in paths.items()}
            finally:
                for handle in handles.values():
                    if not handle.closed:
                        handle.close()
        for path in files_under(FEATURES / "metadata" / "uid"):
            relative = path.relative_to(FEATURES / "metadata" / "uid")
            split = relative.parts[0]
            original = SPLIT / "metadata" / "uid" / relative
            if not original.is_file() or not pq.read_table(path).equals(pq.read_table(original)):
                issues["uid_sidecar_mismatch"] += 1
            file_rows.append({"split": split, "role": "uid_sidecar", "path": str(path.relative_to(PROJECT)).replace("\\", "/"),
                              "bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows})
            stats[split]["bytes"]["uid_sidecar"] += path.stat().st_size
        expected_uid = {path.relative_to(SPLIT / "metadata" / "uid")
                        for path in files_under(SPLIT / "metadata" / "uid")}
        actual_uid = {path.relative_to(FEATURES / "metadata" / "uid")
                      for path in files_under(FEATURES / "metadata" / "uid")}
        if expected_uid != actual_uid:
            issues["uid_sidecar_file_set_mismatch"] += len(expected_uid ^ actual_uid)
        for name in rules["numeric"]:
            minimum = fitted["minimum"].get(name)
            if minimum is None or rules["numeric_fill"][name] != minimum - 1:
                issues["train_numeric_rule_mismatch"] += 1
        for name in rules["categorical"]:
            if sorted(fitted["categories"][name]) != rules["categorical_vocabulary"][name]:
                issues["train_vocabulary_mismatch"] += 1
        if not stats["train"]["max_ts"] < stats["validation"]["min_ts"] < stats["validation"]["max_ts"] < stats["test"]["min_ts"]:
            issues["time_order"] += 1
        with (DATA / "reports" / "splitting" / "split_boundaries.csv").open(encoding="utf-8-sig", newline="") as handle:
            boundaries = {row["boundary"]: int(row["ts"]) for row in csv.DictReader(handle)}
        if not (stats["train"]["max_ts"] <= boundaries["train_end"] < stats["validation"]["min_ts"]
                and stats["validation"]["max_ts"] <= boundaries["validation_end"] < stats["test"]["min_ts"]):
            issues["split_boundaries"] += 1
        for name in SPLITS:
            attacks = set(stats[name]["types"]) - {"normal"}
            if attacks != EXPECTED_ATTACKS[name]:
                issues["attack_protocol"] += 1
        positive_issues = {key: int(value) for key, value in issues.items() if value > 0}
        audit_status = "Đạt" if not positive_issues and duplicates["source"]["cross_split_groups"] == 0 else "Chưa đạt"
        result = {"run_id": run_id, "audited_at": now(), "status": audit_status,
                  "rules_fitted_on": rules.get("fitted_on"),
                  "feature_columns": rules["feature_columns"], "feature_schema": feature_stats["schema"],
                  "feature_stats": feature_stats["values"], "issues": positive_issues, "duplicates": duplicates,
                  "splits": {name: {**item, "labels": dict(item["labels"]), "types": dict(item["types"]),
                                    "bytes": dict(item["bytes"])} for name, item in stats.items()}}
        with (report / "file_sizes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "role", "path", "bytes", "rows"])
            writer.writeheader()
            writer.writerows(file_rows)
        (report / "audit_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "PASS" if audit_status == "Đạt" else "FAIL"
        log.write(f"{now()} END independent_final_audit; rows={total_rows}; issues={positive_issues}; status={status}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        summary = run_audit(args.run_id)
        print(f"AUDIT COMPLETE: {summary['run_id']}; rows={sum(s['rows'] for s in summary['splits'].values())}; issues={summary['issues']}")
        if summary["status"] != "Đạt":
            raise SystemExit(1)
    except Exception as error:
        report = REPORT_OUTPUT_DIR / args.run_id
        if report.is_dir():
            with (report / "execution_log.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} END independent_final_audit; status=FAIL; reason={type(error).__name__}: {error}\n")
        raise
