"""Full-data analysis for the ToN-IoT Network dataset.

This module only analyzes the source CSV files.  It does not clean, transform,
split, or train on the data.  CSV chunks are discarded after each iteration;
temporary binary arrays are used for duplicate and timestamp counts that must span chunks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import heapq
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    from reader import (
        ATTACK_LABELS,
        CHUNK_SIZE,
        EXPECTED_FILE_COUNT,
        NORMAL_LABELS,
        detect_label_columns,
        discover_network_files,
    )
except ModuleNotFoundError:  # Support both script execution and package imports.
    from .reader import (
        ATTACK_LABELS,
        CHUNK_SIZE,
        EXPECTED_FILE_COUNT,
        NORMAL_LABELS,
        detect_label_columns,
        discover_network_files,
    )


REPORT_DIRECTORY = Path(__file__).resolve().parent / "reports" / "analysis"
MISSING_NULL_TOKENS = {
    "<na>",
    "n/a",
    "na",
    "nan",
    "nat",
    "none",
    "null",
}
MISSING_PLACEHOLDER_TOKENS = {
    "-",
    "?",
    "missing",
    "not available",
    "not_available",
    "unknown",
}
CONNECTION_COLUMNS = ["src_ip", "src_port", "dst_ip", "dst_port", "proto", "service"]
IP_RANGE = {f"192.168.1.{number}" for number in range(20, 40)}
GAP_THRESHOLD_SECONDS = 3_600
MAX_EXAMPLES = 5


def normalized_series(series: pd.Series) -> pd.Series:
    """Normalize string values while retaining a marker for missing-like cells."""

    values = series.astype("string").str.strip().str.lower()
    missing = (
        values.isna()
        | values.eq("")
        | values.isin(MISSING_NULL_TOKENS)
        | values.isin(MISSING_PLACEHOLDER_TOKENS)
    )
    return values.mask(missing, "<MISSING>")


def label_state_series(series: pd.Series) -> pd.Series:
    """Convert binary labels to normal, attack, or invalid states."""

    values = normalized_series(series)
    states = pd.Series("invalid", index=series.index, dtype="string")
    states = states.mask(values.isin(NORMAL_LABELS), "normal")
    states = states.mask(values.isin(ATTACK_LABELS), "attack")
    return states


def missing_kind(series: pd.Series) -> dict[str, int]:
    """Count blank, null-token, and placeholder cells separately."""

    values = series.astype("string").str.strip().str.lower()
    blank = values.isna() | values.eq("")
    null_tokens = (~blank) & values.isin(MISSING_NULL_TOKENS)
    placeholders = (~blank) & values.isin(MISSING_PLACEHOLDER_TOKENS)
    return {
        "blank": int(blank.sum()),
        "null_token": int(null_tokens.sum()),
        "placeholder": int(placeholders.sum()),
    }


def make_time_summary() -> dict[str, object]:
    """Create counters for one file or for the whole dataset."""

    return {
        "rows": 0,
        "valid_timestamps": 0,
        "invalid_timestamps": 0,
        "min_ts": None,
        "max_ts": None,
        "out_of_order": 0,
        "last_ts": None,
        "duplicate_timestamp_groups": 0,
        "duplicate_timestamp_extra_rows": 0,
        "gaps_over_threshold": 0,
        "largest_gap_seconds": 0.0,
    }


def make_numeric_issue() -> dict[str, object]:
    """Create counters for one numeric column in one file."""

    return {
        "checked_values": 0,
        "invalid_numeric": 0,
        "negative_values": 0,
        "infinite_values": 0,
        "out_of_range_ports": 0,
        "min_value": None,
        "max_value": None,
        "invalid_examples": [],
        "negative_examples": [],
        "out_of_range_examples": [],
    }


def make_state() -> dict[str, object]:
    """Create the bounded in-memory state used during the streaming pass."""

    return {
        "total_rows": 0,
        "rows_by_file": Counter(),
        "columns_by_file": {},
        "dtypes_by_file": defaultdict(lambda: defaultdict(set)),
        "schema_warnings": [],
        "missing_by_file": defaultdict(lambda: defaultdict(lambda: Counter())),
        "missing_overall": defaultdict(Counter),
        "numeric_issues": defaultdict(lambda: defaultdict(make_numeric_issue)),
        "numeric_candidates": set(),
        "time_by_file": {},
        "time_overall": make_time_summary(),
        "attack_by_file_type": Counter(),
        "attack_by_day_type": Counter(),
        "label_by_file": Counter(),
        "label_by_day": Counter(),
        "attack_stats_by_file": {},
        "attack_stats_overall": {},
        "source_ip_counts": Counter(),
        "destination_ip_counts": Counter(),
        "source_ip_labels": defaultdict(Counter),
        "source_ip_types": defaultdict(Counter),
        "range_ip_stats": defaultdict(lambda: Counter()),
        "feature_values": defaultdict(set),
        "feature_many_values": set(),
        "feature_non_missing": Counter(),
        "feature_missing_like": Counter(),
    }


def to_datetime_utc(timestamp: object) -> str:
    """Format a Unix timestamp as UTC without changing the source value."""

    if timestamp is None or pd.isna(timestamp):
        return ""
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def add_examples(target: list[str], values: pd.Series) -> None:
    """Add at most MAX_EXAMPLES distinct values to a bounded example list."""

    for value in values.astype("string").dropna().drop_duplicates().tolist():
        if str(value) not in target:
            target.append(str(value))
        if len(target) >= MAX_EXAMPLES:
            break


def update_missing_statistics(state: dict[str, object], file_name: str, chunk: pd.DataFrame) -> None:
    """Update missing-value counters for every column in one chunk."""

    for column in chunk.columns:
        counts = missing_kind(chunk[column])
        file_counts = state["missing_by_file"][file_name][column]
        overall_counts = state["missing_overall"][column]
        for key, value in counts.items():
            file_counts[key] += value
            overall_counts[key] += value
        state["feature_missing_like"][column] += sum(counts.values())
        state["feature_non_missing"][column] += len(chunk) - sum(counts.values())


def update_feature_values(state: dict[str, object], chunk: pd.DataFrame) -> None:
    """Track only enough distinct values to identify constant columns."""

    for column in chunk.columns:
        if column in state["feature_many_values"]:
            continue
        values = normalized_series(chunk[column]).drop_duplicates().tolist()
        known_values = state["feature_values"][column]
        known_values.update(str(value) for value in values[:3])
        if len(known_values) > 2:
            state["feature_many_values"].add(column)
            known_values.clear()


def update_dtype_statistics(state: dict[str, object], file_name: str, chunk: pd.DataFrame) -> None:
    """Record dtypes observed while reading the actual chunks."""

    for column, dtype in chunk.dtypes.items():
        state["dtypes_by_file"][file_name][column].add(str(dtype))


def update_numeric_statistics(
    state: dict[str, object],
    file_name: str,
    chunk: pd.DataFrame,
) -> None:
    """Check numeric conversion, negative, infinite, and port-range values."""

    for column in state["numeric_candidates"]:
        if column not in chunk.columns:
            continue
        issue = state["numeric_issues"][file_name][column]
        raw_values = chunk[column].astype("string").str.strip()
        kinds = missing_kind(chunk[column])
        missing_like = (
            raw_values.isna()
            | raw_values.eq("")
            | raw_values.str.lower().isin(MISSING_NULL_TOKENS)
            | raw_values.str.lower().isin(MISSING_PLACEHOLDER_TOKENS)
        )
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        invalid = (~missing_like) & numeric_values.isna()
        infinite = numeric_values.isin([float("inf"), float("-inf")]) & ~missing_like
        finite = numeric_values.notna() & ~infinite
        negative = finite & (numeric_values < 0)

        issue["checked_values"] += int((~missing_like).sum())
        issue["invalid_numeric"] += int(invalid.sum())
        issue["negative_values"] += int(negative.sum())
        issue["infinite_values"] += int(infinite.sum())
        add_examples(issue["invalid_examples"], raw_values[invalid])
        add_examples(issue["negative_examples"], raw_values[negative])

        finite_values = numeric_values[finite]
        if not finite_values.empty:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
            if issue["min_value"] is None or minimum < issue["min_value"]:
                issue["min_value"] = minimum
            if issue["max_value"] is None or maximum > issue["max_value"]:
                issue["max_value"] = maximum

        if column in {"src_port", "dst_port"}:
            out_of_range = finite & ((numeric_values < 0) | (numeric_values > 65535))
            issue["out_of_range_ports"] += int(out_of_range.sum())
            add_examples(issue["out_of_range_examples"], raw_values[out_of_range])


def update_time_statistics(
    state: dict[str, object],
    file_name: str,
    chunk: pd.DataFrame,
) -> None:
    """Update timestamp range and ordering counters."""

    if "ts" not in chunk.columns:
        return

    file_time = state["time_by_file"][file_name]
    overall_time = state["time_overall"]
    values = pd.to_numeric(chunk["ts"], errors="coerce")
    valid = values.notna() & ~values.isin([float("inf"), float("-inf")])
    invalid_count = int((~valid).sum())
    valid_values = values[valid].astype(float).reset_index(drop=True)

    for summary in (file_time, overall_time):
        summary["rows"] += len(chunk)
        summary["valid_timestamps"] += len(valid_values)
        summary["invalid_timestamps"] += invalid_count

    if valid_values.empty:
        return

    for summary in (file_time, overall_time):
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        summary["min_ts"] = minimum if summary["min_ts"] is None else min(summary["min_ts"], minimum)
        summary["max_ts"] = maximum if summary["max_ts"] is None else max(summary["max_ts"], maximum)
        previous = summary["last_ts"]
        if previous is not None and float(valid_values.iloc[0]) < previous:
            summary["out_of_order"] += 1
        summary["out_of_order"] += int((valid_values.iloc[1:].to_numpy() < valid_values.iloc[:-1].to_numpy()).sum())
        summary["last_ts"] = float(valid_values.iloc[-1])



def update_label_and_attack_statistics(
    state: dict[str, object],
    file_name: str,
    chunk: pd.DataFrame,
    label_column: str,
    type_column: str,
) -> None:
    """Update per-file, per-day, and per-type label statistics."""

    type_values = normalized_series(chunk[type_column])
    label_states = label_state_series(chunk[label_column])
    timestamps = pd.to_numeric(chunk["ts"], errors="coerce") if "ts" in chunk else pd.Series(index=chunk.index, dtype=float)
    days = pd.to_datetime(timestamps, unit="s", errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    days = days.fillna("<INVALID>")

    for value, count in type_values.value_counts().items():
        state["attack_by_file_type"][(file_name, str(value))] += int(count)
    for value, count in pd.DataFrame({"day": days, "type": type_values}).value_counts().items():
        state["attack_by_day_type"][(str(value[0]), str(value[1]))] += int(count)
    for value, count in label_states.value_counts().items():
        state["label_by_file"][(file_name, str(value))] += int(count)
    for value, count in pd.DataFrame({"day": days, "label": label_states}).value_counts().items():
        state["label_by_day"][(str(value[0]), str(value[1]))] += int(count)

    update_interval_statistics(
        state["attack_stats_by_file"][file_name],
        type_values,
        timestamps,
        days,
    )
    update_interval_statistics(state["attack_stats_overall"], type_values, timestamps, days)


def update_interval_statistics(
    target: dict[str, dict[str, object]],
    type_values: pd.Series,
    timestamps: pd.Series,
    days: pd.Series,
) -> None:
    """Track count, time interval, and days for each attack type."""

    for attack_type, indices in type_values.groupby(type_values).groups.items():
        key = str(attack_type)
        subset_timestamps = pd.to_numeric(timestamps.loc[indices], errors="coerce").dropna()
        subset_days = set(days.loc[indices].astype(str))
        summary = target.setdefault(
            key,
            {"count": 0, "first_ts": None, "last_ts": None, "days": set()},
        )
        summary["count"] += len(indices)
        if not subset_timestamps.empty:
            minimum = float(subset_timestamps.min())
            maximum = float(subset_timestamps.max())
            summary["first_ts"] = minimum if summary["first_ts"] is None else min(summary["first_ts"], minimum)
            summary["last_ts"] = maximum if summary["last_ts"] is None else max(summary["last_ts"], maximum)
        summary["days"].update(subset_days)


def update_ip_statistics(state: dict[str, object], chunk: pd.DataFrame, label_column: str, type_column: str) -> None:
    """Update source/destination IP counts and label/type relationships."""

    if "src_ip" not in chunk.columns or "dst_ip" not in chunk.columns:
        return
    source_ips = normalized_series(chunk["src_ip"])
    destination_ips = normalized_series(chunk["dst_ip"])
    labels = label_state_series(chunk[label_column])
    types = normalized_series(chunk[type_column])

    source_valid = source_ips[source_ips != "<MISSING>"]
    destination_valid = destination_ips[destination_ips != "<MISSING>"]
    state["source_ip_counts"].update(source_valid.value_counts().to_dict())
    state["destination_ip_counts"].update(destination_valid.value_counts().to_dict())

    source_relationships = pd.DataFrame({"ip": source_ips, "label": labels, "type": types})
    source_relationships = source_relationships[source_relationships["ip"] != "<MISSING>"]
    for (ip, label), count in source_relationships.groupby(["ip", "label"]).size().items():
        state["source_ip_labels"][str(ip)][str(label)] += int(count)
    for (ip, attack_type), count in source_relationships.groupby(["ip", "type"]).size().items():
        state["source_ip_types"][str(ip)][str(attack_type)] += int(count)

    for ip in IP_RANGE:
        source_mask = source_ips.eq(ip)
        destination_mask = destination_ips.eq(ip)
        if source_mask.any():
            state["range_ip_stats"][ip]["source_rows"] += int(source_mask.sum())
            state["range_ip_stats"][ip].update(
                {f"source_{key}": int(value) for key, value in labels[source_mask].value_counts().items()}
            )
            state["range_ip_stats"][ip].update(
                {f"source_type_{key}": int(value) for key, value in types[source_mask].value_counts().items()}
            )
        if destination_mask.any():
            state["range_ip_stats"][ip]["destination_rows"] += int(destination_mask.sum())


def append_array(path: Path, values: np.ndarray, dtype: str | np.dtype) -> None:
    """Append a typed array to a temporary binary file."""

    if len(values) == 0:
        return
    array = np.asarray(values, dtype=dtype)
    with path.open("ab") as binary_file:
        array.tofile(binary_file)


def sorted_binary_array(path: Path, dtype: str | np.dtype) -> np.ndarray:
    """Memory-map a temporary binary array and sort it in place."""

    item_size = np.dtype(dtype).itemsize
    count = path.stat().st_size // item_size if path.exists() else 0
    if count == 0:
        return np.empty(0, dtype=dtype)
    array = np.memmap(path, dtype=dtype, mode="r+", shape=(count,))
    if array.dtype.names:
        array.sort(order="key")
    else:
        array.sort()
    return array


def row_hash_array(frame: pd.DataFrame) -> np.ndarray:
    """Hash all parsed values in a chunk as uint64 values."""

    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)


def connection_array(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Build sortable connection-key/timestamp records for one chunk."""

    keys = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    timestamps = pd.to_numeric(frame["ts"], errors="coerce").to_numpy(dtype=np.float64)
    values = np.empty(len(frame), dtype=[("key", "u8"), ("ts", "f8")])
    values["key"] = keys
    values["ts"] = timestamps
    return values


def uid_hash_array(frame: pd.DataFrame) -> np.ndarray:
    """Hash non-missing UID values for full-data duplicate checking."""

    values = normalized_series(frame["uid"])
    values = values[values != "<MISSING>"]
    if values.empty:
        return np.empty(0, dtype=np.uint64)
    return pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)


def count_sorted_runs(values: np.ndarray) -> tuple[int, int]:
    """Return duplicate groups and extra rows in one sorted array."""

    if len(values) == 0:
        return 0, 0
    change_points = np.flatnonzero(values[1:] != values[:-1]) + 1
    run_lengths = np.diff(np.concatenate(([0], change_points, [len(values)])))
    duplicate_runs = run_lengths[run_lengths > 1]
    return int(len(duplicate_runs)), int((duplicate_runs - 1).sum())


def merged_hash_duplicate_summary(arrays: list[np.ndarray]) -> dict[str, int]:
    """Count duplicate hash groups while merging sorted per-file arrays."""

    def tagged_values(array: np.ndarray, file_index: int):
        for value in array:
            yield int(value), file_index

    iterators = [tagged_values(array, file_index) for file_index, array in enumerate(arrays)]
    current_value = None
    current_count = 0
    current_files: set[int] = set()
    groups = extra_rows = cross_groups = cross_extra_rows = 0

    def finish_group() -> tuple[int, int, int, int]:
        if current_count <= 1:
            return 0, 0, 0, 0
        cross_file = len(current_files) > 1
        return 1, current_count - 1, int(cross_file), current_count - 1 if cross_file else 0

    for value, file_index in heapq.merge(*iterators):
        if current_value is not None and value != current_value:
            group, extra, cross_group, cross_extra = finish_group()
            groups += group
            extra_rows += extra
            cross_groups += cross_group
            cross_extra_rows += cross_extra
            current_count = 0
            current_files = set()
        current_value = value
        current_count += 1
        current_files.add(file_index)
    group, extra, cross_group, cross_extra = finish_group()
    groups += group
    extra_rows += extra
    cross_groups += cross_group
    cross_extra_rows += cross_extra
    return {
        "global_exact_groups": groups,
        "global_exact_extra": extra_rows,
        "cross_file_groups": cross_groups,
        "cross_file_extra": cross_extra_rows,
    }


def merged_timestamp_summary(arrays: list[np.ndarray]) -> dict[str, object]:
    """Count timestamp duplicates and gaps from sorted per-file arrays."""

    def tagged_values(array: np.ndarray, file_index: int):
        for value in array:
            yield float(value), file_index

    iterators = [tagged_values(array, file_index) for file_index, array in enumerate(arrays)]
    current_value = None
    current_count = 0
    duplicate_groups = duplicate_extra = gaps = 0
    largest_gap = 0.0
    examples: list[tuple[float, float, float]] = []

    def finish_timestamp() -> None:
        nonlocal duplicate_groups, duplicate_extra
        if current_count > 1:
            duplicate_groups += 1
            duplicate_extra += current_count - 1

    for value, _ in heapq.merge(*iterators):
        if current_value is None:
            current_value = value
            current_count = 1
        elif value == current_value:
            current_count += 1
        else:
            finish_timestamp()
            gap = value - current_value
            if gap > GAP_THRESHOLD_SECONDS:
                gaps += 1
                largest_gap = max(largest_gap, gap)
                examples.append((current_value, value, gap))
            current_value = value
            current_count = 1
    if current_value is not None:
        finish_timestamp()

    examples.sort(key=lambda row: row[2], reverse=True)
    return {
        "duplicate_timestamp_groups": duplicate_groups,
        "duplicate_timestamp_extra_rows": duplicate_extra,
        "gaps_over_threshold": gaps,
        "largest_gap_seconds": largest_gap,
        "gap_examples": examples[:10],
    }


def single_timestamp_summary(values: np.ndarray) -> dict[str, object]:
    """Count timestamp duplicates and gaps in one sorted file array."""

    duplicate_groups, duplicate_extra = count_sorted_runs(values)
    if len(values) < 2:
        gaps = 0
        largest_gap = 0.0
        examples = []
    else:
        unique_values = np.unique(values)
        gaps_array = np.diff(unique_values)
        gap_indices = np.flatnonzero(gaps_array > GAP_THRESHOLD_SECONDS)
        gaps = int(len(gap_indices))
        largest_gap = float(gaps_array.max()) if len(gaps_array) else 0.0
        examples = [
            (float(unique_values[index]), float(unique_values[index + 1]), float(gaps_array[index]))
            for index in gap_indices
        ]
        examples.sort(key=lambda row: row[2], reverse=True)
    return {
        "duplicate_timestamp_groups": duplicate_groups,
        "duplicate_timestamp_extra_rows": duplicate_extra,
        "gaps_over_threshold": gaps,
        "largest_gap_seconds": largest_gap,
        "gap_examples": examples[:10],
    }


def close_memmaps(arrays: list[np.ndarray]) -> None:
    """Release Windows file handles held by temporary memory maps."""

    for array in arrays:
        if isinstance(array, np.memmap):
            array._mmap.close()


def merged_connection_summary(arrays: list[np.ndarray]) -> dict[str, int]:
    """Count repeated connection keys that have different valid timestamps."""

    iterators = []
    def tagged_connections(array: np.ndarray, file_index: int):
        for row_index, row in enumerate(array):
            timestamp = float(row["ts"]) if not np.isnan(row["ts"]) else None
            yield int(row["key"]), file_index, row_index, timestamp

    for file_index, array in enumerate(arrays):
        iterators.append(tagged_connections(array, file_index))
    current_key = None
    count = 0
    minimum = maximum = None
    groups = extra_rows = 0

    def finish_group() -> None:
        nonlocal groups, extra_rows
        if count > 1 and minimum is not None and maximum is not None and minimum < maximum:
            groups += 1
            extra_rows += count - 1

    for key, _, _, timestamp in heapq.merge(*iterators):
        if current_key is not None and key != current_key:
            finish_group()
            count = 0
            minimum = maximum = None
        current_key = key
        count += 1
        if timestamp is not None:
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
    if current_key is not None:
        finish_group()
    return {"connection_groups": groups, "connection_extra": extra_rows}


def format_values(values: list[object]) -> str:
    """Format bounded examples for a CSV cell."""

    return json.dumps(values, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """Write a small report table."""

    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def write_missing_report(state: dict[str, object], output: Path, file_names: list[str]) -> None:
    """Write per-file and overall missing-value statistics."""

    rows = []
    for file_name in file_names:
        for column, counts in state["missing_by_file"][file_name].items():
            row_count = state["rows_by_file"][file_name]
            missing_like = sum(counts.values())
            rows.append(
                {
                    "scope": "file",
                    "file": file_name,
                    "column": column,
                    "rows": row_count,
                    **counts,
                    "missing_like": missing_like,
                    "missing_like_pct": round(100 * missing_like / row_count, 6) if row_count else 0,
                }
            )
    for column, counts in state["missing_overall"].items():
        missing_like = sum(counts.values())
        rows.append(
            {
                "scope": "overall",
                "file": "",
                "column": column,
                "rows": state["total_rows"],
                **counts,
                "missing_like": missing_like,
                "missing_like_pct": round(100 * missing_like / state["total_rows"], 6)
                if state["total_rows"]
                else 0,
            }
        )
    write_csv(
        output / "missing_values.csv",
        rows,
        [
            "scope",
            "file",
            "column",
            "rows",
            "blank",
            "null_token",
            "placeholder",
            "missing_like",
            "missing_like_pct",
        ],
    )


def write_duplicate_report(
    state: dict[str, object],
    output: Path,
    file_names: list[str],
    exact_by_file: dict[str, tuple[int, int]],
    duplicate_summary: dict[str, int],
) -> dict[str, int]:
    """Write exact duplicate, repeated connection, and UID reports."""

    rows = []
    for file_name in file_names:
        groups, extra_rows = exact_by_file[file_name]
        rows.extend(
            [
                {"scope": file_name, "metric": "exact_duplicate_groups", "count": groups},
                {"scope": file_name, "metric": "exact_duplicate_extra_rows", "count": extra_rows},
            ]
        )

    global_groups = duplicate_summary["global_exact_groups"]
    global_extra = duplicate_summary["global_exact_extra"]
    cross_groups = duplicate_summary["cross_file_groups"]
    cross_extra = duplicate_summary["cross_file_extra"]
    connection_groups = duplicate_summary["connection_groups"]
    connection_extra = duplicate_summary["connection_extra"]
    uid_groups = duplicate_summary["uid_groups"]
    uid_extra = duplicate_summary["uid_extra"]
    rows.extend(
        [
            {"scope": "all_files", "metric": "exact_duplicate_groups", "count": int(global_groups)},
            {"scope": "all_files", "metric": "exact_duplicate_extra_rows", "count": int(global_extra)},
            {"scope": "multiple_files", "metric": "exact_duplicate_groups", "count": int(cross_groups)},
            {"scope": "multiple_files", "metric": "exact_duplicate_extra_rows", "count": int(cross_extra)},
            {"scope": "all_files", "metric": "repeated_connection_groups", "count": int(connection_groups)},
            {"scope": "all_files", "metric": "repeated_connection_extra_rows", "count": int(connection_extra)},
            {"scope": "all_files", "metric": "duplicate_uid_groups", "count": int(uid_groups)},
            {"scope": "all_files", "metric": "duplicate_uid_extra_rows", "count": int(uid_extra)},
        ]
    )
    write_csv(output / "duplicate_analysis.csv", rows, ["scope", "metric", "count"])
    return {
        "global_exact_groups": int(global_groups),
        "global_exact_extra": int(global_extra),
        "cross_file_groups": int(cross_groups),
        "cross_file_extra": int(cross_extra),
        "connection_groups": int(connection_groups),
        "connection_extra": int(connection_extra),
        "uid_groups": int(uid_groups),
        "uid_extra": int(uid_extra),
    }


def write_data_type_report(state: dict[str, object], output: Path, file_names: list[str]) -> None:
    """Write observed dtypes and numeric anomalies for every file/column."""

    rows = []
    for file_name in file_names:
        columns = state["columns_by_file"][file_name]
        for column in columns:
            issue = state["numeric_issues"][file_name].get(column, make_numeric_issue())
            rows.append(
                {
                    "file": file_name,
                    "column": column,
                    "observed_dtypes": "/".join(sorted(state["dtypes_by_file"][file_name][column])),
                    "numeric_candidate": column in state["numeric_candidates"],
                    "checked_values": issue["checked_values"],
                    "invalid_numeric": issue["invalid_numeric"],
                    "negative_values": issue["negative_values"],
                    "infinite_values": issue["infinite_values"],
                    "out_of_range_ports": issue["out_of_range_ports"],
                    "min_value": issue["min_value"],
                    "max_value": issue["max_value"],
                    "invalid_examples": format_values(issue["invalid_examples"]),
                    "negative_examples": format_values(issue["negative_examples"]),
                    "out_of_range_examples": format_values(issue["out_of_range_examples"]),
                }
            )
    write_csv(
        output / "data_type_issues.csv",
        rows,
        [
            "file",
            "column",
            "observed_dtypes",
            "numeric_candidate",
            "checked_values",
            "invalid_numeric",
            "negative_values",
            "infinite_values",
            "out_of_range_ports",
            "min_value",
            "max_value",
            "invalid_examples",
            "negative_examples",
            "out_of_range_examples",
        ],
    )


def write_timestamp_report(
    state: dict[str, object],
    output: Path,
    file_names: list[str],
    overall_timestamp_summary: dict[str, object],
) -> list[dict[str, object]]:
    """Write timestamp summaries, file overlaps, and largest time gaps."""

    rows = []
    for file_name in file_names + ["OVERALL"]:
        summary = state["time_overall"] if file_name == "OVERALL" else state["time_by_file"][file_name]
        if file_name == "OVERALL":
            state["time_overall"].update(overall_timestamp_summary)
            duplicate_groups = overall_timestamp_summary["duplicate_timestamp_groups"]
            duplicate_extra = overall_timestamp_summary["duplicate_timestamp_extra_rows"]
            gaps = overall_timestamp_summary["gaps_over_threshold"]
            largest = overall_timestamp_summary["largest_gap_seconds"]
            examples = overall_timestamp_summary["gap_examples"]
        else:
            duplicate_groups = summary["duplicate_timestamp_groups"]
            duplicate_extra = summary["duplicate_timestamp_extra_rows"]
            gaps = summary["gaps_over_threshold"]
            largest = summary["largest_gap_seconds"]
            examples = []
        rows.append(
            {
                "scope": file_name,
                "rows": summary["rows"],
                "valid_timestamps": summary["valid_timestamps"],
                "invalid_timestamps": summary["invalid_timestamps"],
                "min_ts": summary["min_ts"],
                "max_ts": summary["max_ts"],
                "min_datetime_utc": to_datetime_utc(summary["min_ts"]),
                "max_datetime_utc": to_datetime_utc(summary["max_ts"]),
                "out_of_order": summary["out_of_order"],
                "duplicate_timestamp_groups": duplicate_groups,
                "duplicate_timestamp_extra_rows": duplicate_extra,
                "gaps_over_1h": gaps,
                "largest_gap_seconds": largest,
            }
        )
        if file_name == "OVERALL":
            gap_rows = [
                {
                    "scope": file_name,
                    "start_ts": start,
                    "end_ts": end,
                    "start_datetime_utc": to_datetime_utc(start),
                    "end_datetime_utc": to_datetime_utc(end),
                    "gap_seconds": gap,
                }
                for start, end, gap in examples
            ]
            write_csv(
                output / "timestamp_gap_examples.csv",
                gap_rows,
                [
                    "scope",
                    "start_ts",
                    "end_ts",
                    "start_datetime_utc",
                    "end_datetime_utc",
                    "gap_seconds",
                ],
            )
    write_csv(
        output / "timestamp_analysis.csv",
        rows,
        [
            "scope",
            "rows",
            "valid_timestamps",
            "invalid_timestamps",
            "min_ts",
            "max_ts",
            "min_datetime_utc",
            "max_datetime_utc",
            "out_of_order",
            "duplicate_timestamp_groups",
            "duplicate_timestamp_extra_rows",
            "gaps_over_1h",
            "largest_gap_seconds",
        ],
    )

    overlap_rows = []
    for first, second in itertools.combinations(file_names, 2):
        first_summary = state["time_by_file"][first]
        second_summary = state["time_by_file"][second]
        if first_summary["min_ts"] is None or second_summary["min_ts"] is None:
            continue
        overlap_start = max(first_summary["min_ts"], second_summary["min_ts"])
        overlap_end = min(first_summary["max_ts"], second_summary["max_ts"])
        if overlap_start <= overlap_end:
            overlap_rows.append(
                {
                    "first_file": first,
                    "second_file": second,
                    "overlap_start_ts": overlap_start,
                    "overlap_end_ts": overlap_end,
                    "overlap_start_datetime_utc": to_datetime_utc(overlap_start),
                    "overlap_end_datetime_utc": to_datetime_utc(overlap_end),
                    "overlap_seconds": overlap_end - overlap_start,
                }
            )
    write_csv(
        output / "timestamp_overlaps.csv",
        overlap_rows,
        [
            "first_file",
            "second_file",
            "overlap_start_ts",
            "overlap_end_ts",
            "overlap_start_datetime_utc",
            "overlap_end_datetime_utc",
            "overlap_seconds",
        ],
    )
    return rows


def write_attack_reports(state: dict[str, object], output: Path) -> None:
    """Write attack-type intervals and per-day distribution tables."""

    rows = []
    for scope, summaries in [("file", state["attack_stats_by_file"]), ("overall", {"OVERALL": state["attack_stats_overall"]})]:
        if scope == "file":
            for file_name, type_summaries in summaries.items():
                for attack_type, summary in sorted(type_summaries.items()):
                    rows.append(
                        {
                            "scope": scope,
                            "file": file_name,
                            "type": attack_type,
                            "count": summary["count"],
                            "first_ts": summary["first_ts"],
                            "last_ts": summary["last_ts"],
                            "first_datetime_utc": to_datetime_utc(summary["first_ts"]),
                            "last_datetime_utc": to_datetime_utc(summary["last_ts"]),
                            "days_present": len(summary["days"]),
                            "days": ";".join(sorted(summary["days"])),
                        }
                    )
        else:
            for attack_type, summary in sorted(summaries["OVERALL"].items()):
                rows.append(
                    {
                        "scope": scope,
                        "file": "",
                        "type": attack_type,
                        "count": summary["count"],
                        "first_ts": summary["first_ts"],
                        "last_ts": summary["last_ts"],
                        "first_datetime_utc": to_datetime_utc(summary["first_ts"]),
                        "last_datetime_utc": to_datetime_utc(summary["last_ts"]),
                        "days_present": len(summary["days"]),
                        "days": ";".join(sorted(summary["days"])),
                    }
                )
    write_csv(
        output / "attack_distribution.csv",
        rows,
        [
            "scope",
            "file",
            "type",
            "count",
            "first_ts",
            "last_ts",
            "first_datetime_utc",
            "last_datetime_utc",
            "days_present",
            "days",
        ],
    )

    day_rows = [
        {"day_utc": day, "type": attack_type, "count": count}
        for (day, attack_type), count in sorted(state["attack_by_day_type"].items())
    ]
    write_csv(output / "attack_distribution_by_day.csv", day_rows, ["day_utc", "type", "count"])

    label_file_rows = [
        {"file": file_name, "label_state": label, "count": count}
        for (file_name, label), count in sorted(state["label_by_file"].items())
    ]
    write_csv(output / "label_distribution_by_file.csv", label_file_rows, ["file", "label_state", "count"])
    label_day_rows = [
        {"day_utc": day, "label_state": label, "count": count}
        for (day, label), count in sorted(state["label_by_day"].items())
    ]
    write_csv(output / "label_distribution_by_day.csv", label_day_rows, ["day_utc", "label_state", "count"])

    overlap_rows = []
    types = sorted(attack_type for attack_type in state["attack_stats_overall"] if attack_type != "normal")
    for first_type, second_type in itertools.combinations(types, 2):
        first = state["attack_stats_overall"][first_type]
        second = state["attack_stats_overall"][second_type]
        if first["first_ts"] is None or second["first_ts"] is None:
            continue
        overlap_start = max(first["first_ts"], second["first_ts"])
        overlap_end = min(first["last_ts"], second["last_ts"])
        if overlap_start <= overlap_end:
            overlap_rows.append(
                {
                    "first_type": first_type,
                    "second_type": second_type,
                    "overlap_start_ts": overlap_start,
                    "overlap_end_ts": overlap_end,
                    "overlap_start_datetime_utc": to_datetime_utc(overlap_start),
                    "overlap_end_datetime_utc": to_datetime_utc(overlap_end),
                    "overlap_seconds": overlap_end - overlap_start,
                }
            )
    write_csv(
        output / "attack_type_overlaps.csv",
        overlap_rows,
        [
            "first_type",
            "second_type",
            "overlap_start_ts",
            "overlap_end_ts",
            "overlap_start_datetime_utc",
            "overlap_end_datetime_utc",
            "overlap_seconds",
        ],
    )


def write_feature_report(
    state: dict[str, object],
    output: Path,
    file_names: list[str],
    label_column: str,
    type_column: str,
) -> None:
    """Write a preliminary feature/identifier/label classification."""

    all_columns = []
    for file_name in file_names:
        for column in state["columns_by_file"][file_name]:
            if column not in all_columns:
                all_columns.append(column)

    rows = []
    for column in all_columns:
        if column in {label_column, type_column}:
            role = "label"
            ml_use = "label_only"
            gnn_use = "retain_for_graph_context"
        elif column == "ts":
            role = "time"
            ml_use = "review_before_feature_use"
            gnn_use = "retain_for_temporal_context"
        elif column in {"src_ip", "dst_ip", "src_port", "dst_port", "uid"}:
            role = "identifier_or_connection"
            ml_use = "review_for_leakage"
            gnn_use = "retain_for_graph_context"
        elif column in state["numeric_candidates"]:
            role = "numeric"
            ml_use = "candidate_feature"
            gnn_use = "retain_if_relevant"
        else:
            role = "categorical_text"
            ml_use = "candidate_feature_after_encoding"
            gnn_use = "retain_if_relevant"

        distinct_values = state["feature_values"].get(column, set())
        if column in state["feature_many_values"]:
            distinct_status = "2+"
        else:
            distinct_status = str(len(distinct_values))
        missing_like = state["feature_missing_like"][column]
        non_missing = state["feature_non_missing"][column]
        rows.append(
            {
                "column": column,
                "role": role,
                "files_present": sum(column in state["columns_by_file"][file] for file in file_names),
                "observed_dtypes": "/".join(
                    sorted(
                        {
                            dtype
                            for file_name in file_names
                            for dtype in state["dtypes_by_file"][file_name].get(column, set())
                        }
                    )
                ),
                "missing_like": missing_like,
                "non_missing": non_missing,
                "distinct_status": distinct_status,
                "constant_column": distinct_status == "1",
                "ml_use": ml_use,
                "gnn_use": gnn_use,
                "example_values": format_values(sorted(distinct_values)),
            }
        )
    write_csv(
        output / "feature_analysis.csv",
        rows,
        [
            "column",
            "role",
            "files_present",
            "observed_dtypes",
            "missing_like",
            "non_missing",
            "distinct_status",
            "constant_column",
            "ml_use",
            "gnn_use",
            "example_values",
        ],
    )


def write_ip_report(state: dict[str, object], output: Path) -> dict[str, object]:
    """Write source-IP relationships and the requested private-IP range report."""

    all_ips = sorted(set(state["source_ip_counts"]) | set(state["destination_ip_counts"]))
    rows = []
    ranked_sources = {ip: rank for rank, (ip, _) in enumerate(state["source_ip_counts"].most_common(), start=1)}
    for ip in all_ips:
        labels = state["source_ip_labels"].get(ip, Counter())
        types = state["source_ip_types"].get(ip, Counter())
        source_count = state["source_ip_counts"].get(ip, 0)
        attack_count = labels.get("attack", 0)
        rows.append(
            {
                "ip": ip,
                "source_rank": ranked_sources.get(ip, ""),
                "source_rows": source_count,
                "destination_rows": state["destination_ip_counts"].get(ip, 0),
                "source_normal_rows": labels.get("normal", 0),
                "source_attack_rows": attack_count,
                "source_invalid_label_rows": labels.get("invalid", 0),
                "source_attack_rate_pct": round(100 * attack_count / source_count, 6) if source_count else "",
                "source_type_counts": json.dumps(dict(types), ensure_ascii=False, sort_keys=True),
                "in_requested_range": ip in IP_RANGE,
            }
        )
    write_csv(
        output / "ip_analysis.csv",
        rows,
        [
            "ip",
            "source_rank",
            "source_rows",
            "destination_rows",
            "source_normal_rows",
            "source_attack_rows",
            "source_invalid_label_rows",
            "source_attack_rate_pct",
            "source_type_counts",
            "in_requested_range",
        ],
    )

    range_rows = []
    for ip in sorted(IP_RANGE, key=lambda value: int(value.rsplit(".", 1)[1])):
        stats = state["range_ip_stats"][ip]
        range_rows.append({"ip": ip, **dict(stats)})
    write_csv(output / "ip_range_192_168_1_20_39.csv", range_rows, ["ip"] + sorted({key for stats in state["range_ip_stats"].values() for key in stats}))

    top_source_ip, top_source_count = state["source_ip_counts"].most_common(1)[0] if state["source_ip_counts"] else ("", 0)
    return {
        "unique_source_ips": len(state["source_ip_counts"]),
        "unique_destination_ips": len(state["destination_ip_counts"]),
        "top_source_ip": top_source_ip,
        "top_source_count": top_source_count,
        "source_ip_with_highest_attack_rate": max(
            (
                (ip, counts.get("attack", 0) / sum(counts.values()))
                for ip, counts in state["source_ip_labels"].items()
                if sum(counts.values())
            ),
            key=lambda item: item[1],
            default=("", 0),
        ),
    }


def write_summary(
    state: dict[str, object],
    output: Path,
    file_names: list[str],
    label_column: str | None,
    type_column: str | None,
    duplicate_summary: dict[str, int],
    ip_summary: dict[str, object],
    timestamp_rows: list[dict[str, object]],
    discovery_warnings: list[str],
) -> str:
    """Write and return the human-readable analysis summary."""

    overall_time = state["time_overall"]
    type_counts = Counter()
    for (file_name, attack_type), count in state["attack_by_file_type"].items():
        type_counts[attack_type] += count
    normal_count = type_counts.get("normal", 0)
    attack_count = sum(count for attack_type, count in type_counts.items() if attack_type != "normal")
    schema_consistent = not state["schema_warnings"]
    summary_lines = [
        "========== NETWORK DATA ANALYSIS ==========",
        "",
        "Analysis scope: FULL DATASET (all rows in all selected files; no sampling for final statistics)",
        f"Files analyzed: {len(file_names)} / expected {EXPECTED_FILE_COUNT}",
        f"Total rows: {state['total_rows']:,}",
        f"Detected label column: {label_column or '<not found>'}",
        f"Detected attack-type column: {type_column or '<not found>'}",
        "",
        "1. MISSING VALUES",
        "Missing-like values are separated into blank, null-token, and placeholder ('-') counts.",
        f"Total missing-like cells: {sum(sum(counts.values()) for counts in state['missing_overall'].values()):,}",
        "",
        "2. DUPLICATE ANALYSIS",
        "Exact rows are compared by parsed-row hash using temporary disk-backed binary arrays.",
        f"Exact duplicate groups: {duplicate_summary['global_exact_groups']:,}",
        f"Exact duplicate extra rows: {duplicate_summary['global_exact_extra']:,}",
        f"Exact groups spanning multiple files: {duplicate_summary['cross_file_groups']:,}",
        f"Repeated connection groups with different timestamps: {duplicate_summary['connection_groups']:,}",
        f"Duplicate UID groups: {duplicate_summary['uid_groups']:,}",
        "",
        "3. DATA TYPE ISSUES",
        f"Schema consistency: {'PASS' if schema_consistent else 'FAIL'}",
        "Detailed dtype and numeric checks are in data_type_issues.csv.",
        "",
        "4. TIMESTAMP ANALYSIS",
        f"Earliest timestamp: {overall_time['min_ts']} ({to_datetime_utc(overall_time['min_ts'])})",
        f"Latest timestamp: {overall_time['max_ts']} ({to_datetime_utc(overall_time['max_ts'])})",
        f"Invalid timestamps: {overall_time['invalid_timestamps']:,}",
        f"Out-of-order timestamps: {overall_time['out_of_order']:,}",
        f"Gaps over one hour: {overall_time['gaps_over_threshold']:,}",
        "Timestamp dates in reports are UTC representations of the original Unix seconds; source values were not changed.",
        "",
        "5. ATTACK DISTRIBUTION",
        f"Normal: {normal_count:,}",
        f"Attack: {attack_count:,}",
        "Attack types and time intervals are in attack_distribution.csv.",
        "",
        "6. FEATURE ANALYSIS",
        f"Total distinct columns across files: {len({column for columns in state['columns_by_file'].values() for column in columns})}",
        f"Numeric candidates: {len(state['numeric_candidates'])}",
        "Identifier/connection columns are marked for leakage review and retained for possible graph context.",
        "",
        "7. IP ANALYSIS",
        f"Unique source IPs: {ip_summary['unique_source_ips']:,}",
        f"Unique destination IPs: {ip_summary['unique_destination_ips']:,}",
        f"Most frequent source IP: {ip_summary['top_source_ip']} ({ip_summary['top_source_count']:,} rows)",
        f"Highest source-IP attack rate: {ip_summary['source_ip_with_highest_attack_rate'][0]} ({100 * ip_summary['source_ip_with_highest_attack_rate'][1]:.4f}%)",
        "IP-label relationships are correlations for review, not evidence that a model learned IP identity.",
        "",
        "Validation:",
        f"Rows == normal + attack: {state['total_rows']:,} == {normal_count:,} + {attack_count:,} -> {'PASS' if state['total_rows'] == normal_count + attack_count else 'FAIL'}",
        f"Schema consistency: {'PASS' if schema_consistent else 'FAIL'}",
        "",
        "Warnings:",
    ]
    summary_lines.extend(f"- {warning}" for warning in discovery_warnings + state["schema_warnings"])
    summary_lines.extend(
        [
            "",
            f"Reports saved to: {output}",
            "",
            "===========================================",
        ]
    )
    summary = "\n".join(summary_lines) + "\n"
    (output / "analysis_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def prepare_schema(
    files: list[Path],
    state: dict[str, object],
) -> tuple[str | None, str | None, set[str]]:
    """Inspect headers/sample dtypes and identify the label/type columns."""

    first_sample = pd.read_csv(files[0], nrows=1_000, low_memory=False)
    label_column, type_column = detect_label_columns(first_sample)
    reference_columns = tuple(str(column) for column in first_sample.columns)

    for path in files:
        sample = pd.read_csv(path, nrows=100_000, low_memory=False)
        file_name = path.name
        columns = tuple(str(column) for column in sample.columns)
        state["columns_by_file"][file_name] = columns
        state["dtypes_by_file"][file_name] = defaultdict(set)
        for column, dtype in sample.dtypes.items():
            state["dtypes_by_file"][file_name][str(column)].add(str(dtype))
        for column, dtype in sample.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                state["numeric_candidates"].add(str(column))
        if columns != reference_columns:
            missing = [column for column in reference_columns if column not in columns]
            extra = [column for column in columns if column not in reference_columns]
            state["schema_warnings"].append(
                f"{file_name}: schema differs from {files[0].name}; missing={missing}, extra={extra}."
            )
    return label_column, type_column, state["numeric_candidates"]


def analyze_file(
    path: Path,
    state: dict[str, object],
    label_column: str | None,
    type_column: str | None,
    connection_columns: list[str],
    external_paths: dict[str, Path],
) -> tuple[int, int]:
    """Analyze one file in chunks and write bounded binary intermediates."""

    file_name = path.name
    state["time_by_file"][file_name] = make_time_summary()
    state["attack_stats_by_file"][file_name] = {}

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
        on_bad_lines="error",
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        print(f"  {file_name}: chunk {chunk_number} ({len(chunk):,} rows)", flush=True)
        state["total_rows"] += len(chunk)
        state["rows_by_file"][file_name] += len(chunk)
        update_dtype_statistics(state, file_name, chunk)
        update_missing_statistics(state, file_name, chunk)
        update_feature_values(state, chunk)
        update_numeric_statistics(state, file_name, chunk)
        update_time_statistics(state, file_name, chunk)

        timestamps = pd.to_numeric(chunk["ts"], errors="coerce") if "ts" in chunk else pd.Series(dtype=float)
        valid_timestamps = timestamps.notna() & ~timestamps.isin([float("inf"), float("-inf")])
        append_array(
            external_paths["timestamps"],
            timestamps[valid_timestamps].to_numpy(dtype=np.float64),
            np.float64,
        )

        if label_column in chunk.columns and type_column in chunk.columns:
            update_label_and_attack_statistics(state, file_name, chunk, label_column, type_column)
            update_ip_statistics(state, chunk, label_column, type_column)

        append_array(external_paths["rows"], row_hash_array(chunk), np.uint64)
        if connection_columns:
            append_array(external_paths["connections"], connection_array(chunk, connection_columns), external_paths["connection_dtype"])
        if "uid" in chunk.columns:
            append_array(external_paths["uids"], uid_hash_array(chunk), np.uint64)

    return 0, 0


def finalize_external_analysis(
    files: list[Path],
    external_paths_by_file: dict[str, dict[str, object]],
    state: dict[str, object],
) -> tuple[dict[str, tuple[int, int]], dict[str, int], dict[str, object]]:
    """Sort temporary arrays and calculate full-data duplicate/time summaries."""

    exact_by_file = {}
    row_arrays = []
    timestamp_arrays = []
    connection_arrays = []
    uid_arrays = []

    for path in files:
        file_name = path.name
        paths = external_paths_by_file[file_name]
        rows = sorted_binary_array(paths["rows"], np.uint64)
        timestamps = sorted_binary_array(paths["timestamps"], np.float64)
        connections = sorted_binary_array(paths["connections"], paths["connection_dtype"])
        uids = sorted_binary_array(paths["uids"], np.uint64)
        row_arrays.append(rows)
        timestamp_arrays.append(timestamps)
        connection_arrays.append(connections)
        uid_arrays.append(uids)
        exact_by_file[file_name] = count_sorted_runs(rows)
        state["time_by_file"][file_name].update(single_timestamp_summary(timestamps))

    exact_summary = merged_hash_duplicate_summary(row_arrays)
    timestamp_summary = merged_timestamp_summary(timestamp_arrays)
    connection_summary = merged_connection_summary(connection_arrays)
    uid_summary = merged_hash_duplicate_summary(uid_arrays)
    duplicate_summary = {
        **exact_summary,
        **connection_summary,
        "uid_groups": uid_summary["global_exact_groups"],
        "uid_extra": uid_summary["global_exact_extra"],
    }
    close_memmaps(row_arrays + timestamp_arrays + connection_arrays + uid_arrays)
    return exact_by_file, duplicate_summary, timestamp_summary


def main() -> int:
    """Run the full-data analysis and write all reports."""

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_directory, files, discovery_warnings = discover_network_files()
    state = make_state()
    label_column, type_column, _ = prepare_schema(files, state)
    connection_columns = [
        column for column in CONNECTION_COLUMNS if column in state["columns_by_file"][files[0].name]
    ]

    connection_dtype = np.dtype([("key", "u8"), ("ts", "f8")])
    with tempfile.TemporaryDirectory(prefix="network_analysis_", dir=REPORT_DIRECTORY) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        external_paths_by_file = {}
        for path in files:
            file_name = path.name
            external_paths_by_file[file_name] = {
                "rows": temporary_directory / f"{file_name}.rows.bin",
                "timestamps": temporary_directory / f"{file_name}.timestamps.bin",
                "connections": temporary_directory / f"{file_name}.connections.bin",
                "uids": temporary_directory / f"{file_name}.uids.bin",
                "connection_dtype": connection_dtype,
            }

        for path in files:
            print(f"Analyzing {path.name} ...", flush=True)
            analyze_file(
                path,
                state,
                label_column,
                type_column,
                connection_columns,
                external_paths_by_file[path.name],
            )

        exact_by_file, duplicate_summary, timestamp_summary = finalize_external_analysis(
            files,
            external_paths_by_file,
            state,
        )

        write_missing_report(state, REPORT_DIRECTORY, [path.name for path in files])
        duplicate_summary = write_duplicate_report(
            state,
            REPORT_DIRECTORY,
            [path.name for path in files],
            exact_by_file,
            duplicate_summary,
        )
        write_data_type_report(state, REPORT_DIRECTORY, [path.name for path in files])
        timestamp_rows = write_timestamp_report(
            state,
            REPORT_DIRECTORY,
            [path.name for path in files],
            timestamp_summary,
        )
        write_attack_reports(state, REPORT_DIRECTORY)
        write_feature_report(
            state,
            REPORT_DIRECTORY,
            [path.name for path in files],
            label_column or "label",
            type_column or "type",
        )
        ip_summary = write_ip_report(state, REPORT_DIRECTORY)

    summary = write_summary(
        state,
        REPORT_DIRECTORY,
        [path.name for path in files],
        label_column,
        type_column,
        duplicate_summary,
        ip_summary,
        timestamp_rows,
        discovery_warnings,
    )
    print(summary)

    row_counts = Counter()
    for (_, _), count in state["attack_by_file_type"].items():
        row_counts["classified"] += count
    rows_match = row_counts["classified"] == state["total_rows"]
    return 0 if len(files) == EXPECTED_FILE_COUNT and rows_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
