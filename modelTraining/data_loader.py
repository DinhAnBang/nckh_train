"""Validate aligned Parquet parts and build bounded sparse batches."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from .utils import GIB


SPLITS = ("train", "validation", "test")
FORBIDDEN = {"label", "type", "ts", "src_ip", "dst_ip", "record_id",
             "duplicate_key", "source_file", "source_row", "uid"}
MAX_EXACT_FLOAT64_INT = 2**53
EXPECTED_FEATURE_COUNT = 197


def inspect_layout(data_dir):
    """Match every X part with its y and metadata, without loading all rows."""
    root = Path(data_dir).expanduser().resolve()
    rules_path = root / "metadata" / "train_fitted_rules.json"
    if not rules_path.is_file():
        raise FileNotFoundError(f"Train-fitted rules missing: {rules_path}")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    columns = rules["feature_columns"]
    if len(columns) != EXPECTED_FEATURE_COUNT or len(columns) != len(set(columns)) or set(columns) & FORBIDDEN:
        raise ValueError("Expected 197 unique features without label or metadata columns")
    if rules.get("fitted_on") != "train_only":
        raise ValueError("Feature rules were not fitted on train only")
    parts, counts = {}, Counter()
    reference_types = None
    parquet_bytes = 0
    for name in SPLITS:
        x_root = root / name / "X"
        parts[name] = sorted(x_root.rglob("*.parquet"))
        if not parts[name]:
            raise FileNotFoundError(f"No X parts under {x_root}")
        for x_path in parts[name]:
            relative = x_path.relative_to(x_root)
            y_path = root / name / "y" / relative
            metadata_path = root / name / "metadata" / relative
            if not y_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Unpaired X/y/metadata: {relative}")
            x, y, metadata = (pq.ParquetFile(path) for path in (x_path, y_path, metadata_path))
            rows = x.metadata.num_rows
            if rows == 0:
                raise ValueError(f"Empty feature part: {x_path}")
            if rows != y.metadata.num_rows or rows != metadata.metadata.num_rows:
                raise ValueError(f"X/y/metadata row mismatch: {relative}")
            fields = [(field.name, field.type) for field in x.schema_arrow]
            if x.schema_arrow.names != columns or any(not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)) for _, t in fields):
                raise ValueError(f"Non-numeric or unordered features: {x_path}")
            if reference_types is None:
                reference_types = fields
            elif fields != reference_types:
                raise ValueError(f"Feature dtypes differ: {x_path}")
            if y.schema_arrow.names != ["label"] or "type" not in metadata.schema_arrow.names or "record_id" not in metadata.schema_arrow.names:
                raise ValueError(f"Invalid y or metadata schema: {relative}")
            counts[name] += rows
            parquet_bytes += x_path.stat().st_size
    if counts["train"] != rules["train_rows"]:
        raise ValueError("Train row count differs from fitted rules")
    return {"root": root, "parts": parts, "rows": counts, "columns": columns,
            "types": reference_types, "rules_path": rules_path, "x_parquet_bytes": parquet_bytes}


def paired_paths(layout, split, x_path):
    relative = x_path.relative_to(layout["root"] / split / "X")
    return (x_path, layout["root"] / split / "y" / relative,
            layout["root"] / split / "metadata" / relative)


def to_csr(table, columns):
    """Build CSR directly from nonzero columns, avoiding a full dense matrix."""
    row_indices, col_indices, values = [], [], []
    rows = table.num_rows
    for index, name in enumerate(columns):
        column = table.column(name)
        if column.null_count:
            raise ValueError(f"Null feature: {name}")
        array = column.to_numpy(zero_copy_only=False)
        if pa.types.is_boolean(column.type):
            positions = np.flatnonzero(array)
            data = np.ones(len(positions), dtype=np.float64)
        else:
            if pa.types.is_integer(column.type) and np.any((array > MAX_EXACT_FLOAT64_INT) | (array < -MAX_EXACT_FLOAT64_INT)):
                raise ValueError(f"Float64 cannot exactly represent {name}; use a different encoding")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite feature: {name}")
            positions = np.flatnonzero(array)
            data = array[positions].astype(np.float64, copy=False)
        row_indices.append(positions.astype(np.int32, copy=False))
        col_indices.append(np.full(len(positions), index, dtype=np.int32))
        values.append(data)
    matrix = sparse.coo_matrix((np.concatenate(values),
                                (np.concatenate(row_indices), np.concatenate(col_indices))),
                               shape=(rows, len(columns)), dtype=np.float64).tocsr()
    return matrix


def audit_data(layout):
    """Dry-run checks every feature value, binary label and part alignment."""
    labels = Counter()
    nonzero_sample = None
    for name in SPLITS:
        for number, x_path in enumerate(layout["parts"][name], start=1):
            _, y_path, metadata_path = paired_paths(layout, name, x_path)
            x = pq.read_table(x_path)
            y = pq.read_table(y_path, columns=["label"]).column(0).to_numpy()
            metadata = pq.read_table(metadata_path, columns=["record_id", "type"])
            if x.num_rows != len(y) or x.num_rows != metadata.num_rows:
                raise ValueError(f"Part alignment failed: {x_path}")
            if not np.isin(y, [0, 1]).all() or metadata.column("record_id").null_count:
                raise ValueError(f"Invalid label or trace ID: {x_path}")
            kinds = metadata.column("type").to_pylist()
            if any(kind is None or str(kind).strip() == "" for kind in kinds):
                raise ValueError(f"Missing attack type: {x_path}")
            is_normal = np.fromiter((str(kind).strip().lower() == "normal" for kind in kinds),
                                    dtype=bool, count=len(kinds))
            if not np.array_equal(is_normal, y == 0):
                raise ValueError(f"Label and attack type disagree: {x_path}")
            for column in x.columns:
                if column.null_count:
                    raise ValueError(f"Null feature in {x_path}")
                if pa.types.is_floating(column.type) and not bool(pc.all(pc.is_finite(column)).as_py()):
                    raise ValueError(f"Non-finite feature in {x_path}")
                if pa.types.is_integer(column.type):
                    minimum, maximum = pc.min(column).as_py(), pc.max(column).as_py()
                    if minimum < -MAX_EXACT_FLOAT64_INT or maximum > MAX_EXACT_FLOAT64_INT:
                        raise ValueError(f"Float64 cannot exactly represent a feature in {x_path}")
            if nonzero_sample is None:
                nonzero_sample = to_csr(x, layout["columns"]).nnz / x.num_rows
            labels[(name, 0)] += int((y == 0).sum())
            labels[(name, 1)] += int((y == 1).sum())
            if number % 50 == 0:
                print(f"Audited {name}: {number}/{len(layout['parts'][name])} parts", flush=True)
    if any(labels[(name, value)] == 0 for name in SPLITS for value in (0, 1)):
        raise ValueError("At least one split lacks a binary class")
    return labels, nonzero_sample


def memory_estimate(layout, nnz_per_row):
    """Report approximate CSR and dense float64 sizes, not a fit guarantee."""
    train = layout["rows"]["train"]
    validation = layout["rows"]["validation"]
    features = len(layout["columns"])
    def csr_gib(rows):
        return (rows * nnz_per_row * (8 + 4) + (rows + 1) * 4) / GIB
    return {"nnz_per_row_sample": round(nnz_per_row, 2),
            "train_csr_est_gib": round(csr_gib(train), 2),
            "validation_csr_est_gib": round(csr_gib(validation), 2),
            "train_dense_float64_gib": round(train * features * 8 / GIB, 2),
            "validation_dense_float64_gib": round(validation * features * 8 / GIB, 2),
            "x_parquet_gib": round(layout["x_parquet_bytes"] / GIB, 2)}


def require_full_memory(layout, estimate, available_gib, model):
    """Block an unsafe full load; estimator internals can need still more RAM."""
    dense = estimate["train_dense_float64_gib"]
    validation = estimate["validation_dense_float64_gib"] if model != "rf" else 0
    required = max(4 * estimate["train_csr_est_gib"] + 2 * estimate["validation_csr_est_gib"] + 4,
                   (3 if model == "rf" else 2.5) * dense + validation + 4)
    if available_gib < required:
        raise MemoryError(f"Full {model} training requires a conservative RAM budget of about {required:.1f} GiB; available {available_gib:.1f} GiB. Use --dry-run, --smoke-test, or a larger machine. No data was sampled automatically.")
    return round(required, 2)


def balanced_sample(layout, split, limit):
    """Take first rows of each class deterministically; technical tests only."""
    quotas = {0: limit // 2, 1: limit - limit // 2}
    matrices, labels, record_ids, attack_types = [], [], [], []
    for x_path in layout["parts"][split]:
        _, y_path, metadata_path = paired_paths(layout, split, x_path)
        y = pq.read_table(y_path, columns=["label"]).column(0).to_numpy()
        positions = []
        for value in (0, 1):
            take = min(quotas[value], int((y == value).sum()))
            positions.extend(np.flatnonzero(y == value)[:take].tolist())
            quotas[value] -= take
        if positions:
            positions.sort()
            indexes = pa.array(positions, type=pa.int64())
            x = pq.read_table(x_path).take(indexes)
            metadata = pq.read_table(metadata_path, columns=["record_id", "type"]).take(indexes)
            ids = metadata.column("record_id").to_pylist()
            matrices.append(to_csr(x, layout["columns"]))
            labels.append(y[positions])
            record_ids.extend(ids)
            attack_types.extend(metadata.column("type").to_pylist())
        if all(value == 0 for value in quotas.values()):
            break
    if any(quotas.values()):
        raise ValueError(f"Cannot collect both classes for {split}; missing quotas {quotas}")
    return sparse.vstack(matrices, format="csr"), np.concatenate(labels), record_ids, attack_types


def load_full(layout, split):
    """Load one split as CSR; this is not incremental model fitting."""
    matrices, labels = [], []
    for number, x_path in enumerate(layout["parts"][split], start=1):
        _, y_path, _ = paired_paths(layout, split, x_path)
        matrices.append(to_csr(pq.read_table(x_path), layout["columns"]))
        labels.append(pq.read_table(y_path, columns=["label"]).column(0).to_numpy())
        if number % 25 == 0:
            print(f"Loaded {split}: {number}/{len(layout['parts'][split])} parts", flush=True)
    result = sparse.vstack(matrices, format="csr")
    if result.shape[0] != layout["rows"][split]:
        raise ValueError(f"Full {split} row mismatch")
    return result, np.concatenate(labels)


def eval_batches(layout, split, dense=False):
    for x_path in layout["parts"][split]:
        _, y_path, metadata_path = paired_paths(layout, split, x_path)
        x = to_csr(pq.read_table(x_path), layout["columns"])
        if dense:
            x = x.toarray()
        y = pq.read_table(y_path, columns=["label"]).column(0).to_numpy()
        kinds = pq.read_table(metadata_path, columns=["type"]).column(0).to_pylist()
        yield x, y, kinds
