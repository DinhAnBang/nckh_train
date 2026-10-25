"""Fit Network feature rules on train only and apply them to all three splits."""

from collections import Counter
import csv
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .cleaner import BASE, write_parquet


SPLIT = BASE / "outputs" / "split"
STAGING = BASE / "outputs" / "features_staging"
OUTPUT = BASE / "outputs" / "features"
REPORT = BASE / "reports" / "preprocessing"
SPLITS = ("train", "validation", "test")
MAX_CATEGORIES = 64
CARDINALITY_LIMIT = MAX_CATEGORIES + 1
MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}
EXCLUDED_IDENTIFIERS = {
    "label", "type", "ts", "src_ip", "dst_ip", "uid", "record_id",
    "duplicate_key", "source_file", "source_row",
}
EXCLUDED_TEXT = {
    "dns_query": "domain/query can identify a device or attack campaign",
    "http_uri": "URI can contain identifiers, payloads and campaign markers",
    "http_referrer": "URL can identify a host or campaign",
    "http_user_agent": "free text can identify a client or campaign",
    "ssl_subject": "certificate identity",
    "ssl_issuer": "certificate identity",
    "weird_addl": "free-form context can contain identifiers",
}
METADATA = ["record_id", "source_file", "source_row", "duplicate_key", "ts",
            "src_ip", "dst_ip", "src_port", "dst_port", "proto", "service", "type"]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def split_parts(name):
    return sorted((SPLIT / name).rglob("*.parquet"))


def inspect_inputs():
    """Check split status, Parquet schemas and recorded row counts."""
    summary = (BASE / "reports" / "splitting" / "split_summary.md").read_text(encoding="utf-8")
    if "Kết quả cuối: **PASS**" not in summary:
        raise ValueError("Step 4 split is not PASS")
    expected = Counter()
    with (BASE / "reports" / "splitting" / "split_distribution.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            expected[row["split"]] += int(row["count"])
    paths = {name: split_parts(name) for name in SPLITS}
    if any(not paths[name] for name in SPLITS):
        raise FileNotFoundError("A split has no Parquet data")
    schema = pq.ParquetFile(paths["train"][0]).schema_arrow
    actual = Counter()
    for name in SPLITS:
        for path in paths[name]:
            part = pq.ParquetFile(path)
            actual[name] += part.metadata.num_rows
            if [(field.name, field.type) for field in part.schema_arrow] != [(field.name, field.type) for field in schema]:
                raise ValueError(f"Inconsistent split schema: {path}")
    if actual != expected:
        raise ValueError(f"Split row counts disagree with step 4: {actual - expected}")
    if not set(METADATA + ["label"]).issubset(schema.names):
        raise ValueError("Required label or metadata is absent")
    return paths, schema, actual


def scalar_count(value):
    result = value.as_py()
    return 0 if result is None else int(result)


def missing_text(series):
    values = series.astype("string")
    return values.isna() | values.str.strip().str.lower().isin(MISSING_TOKENS)


def fit_train(paths, schema):
    """Learn cardinality, missingness and numeric minima from all train rows."""
    stats = {name: {"null": 0, "dash": 0, "blank": 0, "min": None, "max": None,
                    "categories": set(), "many": False} for name in schema.names}
    numeric = [field.name for field in schema if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)]
    categorical = [field.name for field in schema if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
    for number, path in enumerate(paths, start=1):
        table = pq.read_table(path)
        for name in numeric:
            column = table.column(name)
            item = stats[name]
            item["null"] += column.null_count
            if pa.types.is_floating(column.type) and not bool(pc.all(pc.is_finite(column)).as_py()):
                raise ValueError(f"Non-finite train value in {name}: {path}")
            minimum, maximum = pc.min(column).as_py(), pc.max(column).as_py()
            if minimum is not None:
                item["min"] = minimum if item["min"] is None else min(item["min"], minimum)
                item["max"] = maximum if item["max"] is None else max(item["max"], maximum)
        for name in categorical:
            column = table.column(name)
            item = stats[name]
            item["null"] += column.null_count
            item["dash"] += scalar_count(pc.sum(pc.equal(column, "-")))
            item["blank"] += scalar_count(pc.sum(pc.equal(column, "")))
            if not item["many"]:
                values = pc.unique(column).to_pylist()
                for value in values:
                    if value is None or str(value).strip().lower() in MISSING_TOKENS:
                        continue
                    item["categories"].add(value)
                    if len(item["categories"]) >= CARDINALITY_LIMIT:
                        item["many"] = True
                        item["categories"].clear()
                        break
        if number % 25 == 0:
            print(f"Fitting train rules: {number}/{len(paths)} parts", flush=True)
    return stats


def select_features(schema, stats):
    """Choose fields by source meaning and train-only cardinality."""
    rows = []
    numeric, categorical = [], []
    for field in schema:
        name = field.name
        item = stats[name]
        kind = "numeric" if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) else "categorical"
        if name in EXCLUDED_IDENTIFIERS:
            reason = "label, time, identity or trace metadata"
        elif name in EXCLUDED_TEXT:
            reason = EXCLUDED_TEXT[name]
        elif kind == "numeric" and item["min"] is None:
            reason = "no observed numeric value in train"
        elif kind == "numeric" and item["min"] == item["max"] and item["null"] == 0:
            reason = "constant in train"
        elif kind == "categorical" and item["many"]:
            reason = f"more than {MAX_CATEGORIES} categories in train"
        elif kind == "categorical" and len(item["categories"]) <= 1 and item["null"] + item["blank"] == 0:
            reason = "constant in train"
        else:
            reason = "selected; train-only rule"
            (numeric if kind == "numeric" else categorical).append(name)
        if name in ("src_port", "dst_port") and reason.startswith("selected"):
            reason += "; port values may encode environment, review by ablation"
        rows.append({"column": name, "source_dtype": str(field.type), "role": kind,
                     "selected": reason.startswith("selected"), "train_null": item["null"],
                     "train_dash": item["dash"], "train_blank": item["blank"],
                     "train_cardinality": f">{MAX_CATEGORIES}" if item["many"] else len(item["categories"]) if kind == "categorical" else "continuous",
                     "reason": reason})
    if not numeric or not categorical:
        raise ValueError("No usable numeric or categorical features")
    return numeric, categorical, rows


def make_rules(numeric, categorical, stats, input_rows):
    fills = {}
    vocabularies = {}
    columns = []
    for name in numeric:
        minimum = stats[name]["min"]
        fills[name] = int(minimum) - 1 if isinstance(minimum, int) else float(minimum) - 1.0
        columns.append(name)
        if stats[name]["null"]:
            columns.append(name + "__was_missing")
    for name in categorical:
        vocabularies[name] = sorted(stats[name]["categories"])
        columns += [f"{name}__cat_{index:03}" for index in range(len(vocabularies[name]))]
        columns += [name + "__missing", name + "__unknown"]
    if len(columns) != len(set(columns)):
        raise ValueError("Output feature names collide")
    return {"fitted_on": "train_only", "train_rows": input_rows, "numeric": numeric,
            "categorical": categorical, "numeric_fill": fills,
            "categorical_vocabulary": vocabularies, "feature_columns": columns,
            "no_scaling": True, "missing_categorical": "blank/null-token has own flag; '-' remains a category",
            "unseen_categorical": "unknown flag, existing columns unchanged"}


def transform(frame, rules, counts):
    """Produce one bounded X part using immutable train-fitted rules."""
    output = {}
    for name in rules["numeric"]:
        values = frame[name]
        absent = values.isna()
        if absent.any():
            counts[(name, "numeric_imputed")] += int(absent.sum())
        dtype = np.float64 if pd.api.types.is_float_dtype(values.dtype) else np.int64
        output[name] = values.fillna(rules["numeric_fill"][name]).to_numpy(dtype=dtype)
        indicator = name + "__was_missing"
        if indicator in rules["feature_columns"]:
            output[indicator] = absent.to_numpy(dtype=bool)
    for name in rules["categorical"]:
        values = frame[name].astype("string")
        absent = missing_text(values)
        vocabulary = rules["categorical_vocabulary"][name]
        known = values.isin(vocabulary)
        for index, category in enumerate(vocabulary):
            output[f"{name}__cat_{index:03}"] = values.eq(category).fillna(False).to_numpy(dtype=bool)
        output[name + "__missing"] = absent.to_numpy(dtype=bool)
        unseen = (~absent) & (~known)
        output[name + "__unknown"] = unseen.to_numpy(dtype=bool)
        counts[(name, "unseen")] += int(unseen.sum())
        counts[(name, "missing")] += int(absent.sum())
        counts[(name, "dash")] += int(values.eq("-").sum())
    result = pd.DataFrame(output, columns=rules["feature_columns"])
    return result


def export_split(name, parts, rules, counters):
    """Write aligned X, y and trace metadata parts without joining all rows."""
    selected = rules["numeric"] + rules["categorical"]
    read_columns = list(dict.fromkeys(selected + ["label"] + METADATA))
    for number, path in enumerate(parts, start=1):
        frame = pq.read_table(path, columns=read_columns).to_pandas()
        relative = path.relative_to(SPLIT / name)
        x = transform(frame, rules, counters[name])
        labels = frame["label"]
        if not labels.isin(["0", "1"]).all():
            raise ValueError(f"Invalid binary label in {path}")
        y = pd.DataFrame({"label": labels.astype("uint8")})
        metadata = frame[METADATA].copy()
        write_parquet(x, STAGING / name / "X" / relative)
        write_parquet(y, STAGING / name / "y" / relative)
        write_parquet(metadata, STAGING / name / "metadata" / relative)
        counters[name][("__all__", "rows")] += len(frame)
        counters[name][("__all__", "normal")] += int((y["label"] == 0).sum())
        counters[name][("__all__", "attack")] += int((y["label"] == 1).sum())
        if number % 25 == 0:
            print(f"Transforming {name}: {number}/{len(parts)} parts", flush=True)
    uid_source = SPLIT / "metadata" / "uid" / name
    if uid_source.exists():
        for path in uid_source.rglob("*.parquet"):
            destination = STAGING / "metadata" / "uid" / name / path.relative_to(uid_source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def expected_labels():
    expected = Counter()
    with (BASE / "reports" / "splitting" / "split_distribution.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            expected[(row["split"], row["label"])] += int(row["count"])
    return expected


def validate_outputs(paths, rules, input_rows, root=STAGING):
    """Read all saved X/y parts and check alignment, values and split labels."""
    checks = {}
    observed_rows = Counter()
    observed_labels = Counter()
    schema = None
    valid_schema = valid_numeric = aligned = True
    sample_repeatable = False
    unseen_counts = Counter()
    for name in SPLITS:
        for path in paths[name]:
            relative = path.relative_to(SPLIT / name)
            x_path = root / name / "X" / relative
            y_path = root / name / "y" / relative
            meta_path = root / name / "metadata" / relative
            x_file, y_file, meta_file = pq.ParquetFile(x_path), pq.ParquetFile(y_path), pq.ParquetFile(meta_path)
            length = x_file.metadata.num_rows
            observed_rows[name] += length
            aligned &= length == y_file.metadata.num_rows == meta_file.metadata.num_rows
            x_schema = x_file.schema_arrow
            if schema is None:
                schema = [(field.name, field.type) for field in x_schema]
            valid_schema &= x_schema.names == rules["feature_columns"] and [(field.name, field.type) for field in x_schema] == schema
            numeric_table = pq.read_table(x_path, columns=rules["numeric"])
            for column in numeric_table.columns:
                valid_numeric &= column.null_count == 0
                if pa.types.is_floating(column.type):
                    valid_numeric &= bool(pc.all(pc.is_finite(column)).as_py())
            label_values = pq.read_table(y_path, columns=["label"]).column(0).to_numpy()
            observed_labels[(name, "0")] += int((label_values == 0).sum())
            observed_labels[(name, "1")] += int((label_values == 1).sum())
            original = pq.read_table(path, columns=["label", "record_id"])
            saved_ids = pq.read_table(meta_path, columns=["record_id"])
            aligned &= original.column("record_id").equals(saved_ids.column("record_id"))
            aligned &= bool(pc.all(pc.equal(pc.cast(original.column("label"), pa.uint8()), pa.array(label_values))).as_py())
            unknown_columns = [column + "__unknown" for column in rules["categorical"]]
            flags = pq.read_table(x_path, columns=unknown_columns)
            for column, flag in zip(unknown_columns, flags.columns):
                unseen_counts[(name, column.removesuffix("__unknown"))] += scalar_count(pc.sum(flag))
            if not sample_repeatable:
                sample = pq.read_table(path, columns=list(dict.fromkeys(rules["numeric"] + rules["categorical"]))).to_pandas().head(10)
                first = transform(sample, rules, Counter())
                saved = pq.read_table(x_path).slice(0, 10).to_pandas()
                sample_repeatable = first.equals(transform(sample, rules, Counter())) and first.equals(saved)
    checks["row_preservation"] = observed_rows == input_rows
    checks["label_preservation"] = observed_labels == expected_labels()
    checks["alignment"] = aligned
    checks["feature_consistency"] = valid_schema
    checks["numeric_validation"] = valid_numeric
    checks["repeatable_transform"] = sample_repeatable
    checks["no_banned_columns"] = not (set(rules["feature_columns"]) & EXCLUDED_IDENTIFIERS)
    checks["train_only_rules"] = rules["fitted_on"] == "train_only" and rules["train_rows"] == input_rows["train"]
    uid_input = sum(pq.ParquetFile(path).metadata.num_rows for path in (SPLIT / "metadata" / "uid").rglob("*.parquet"))
    uid_output = sum(pq.ParquetFile(path).metadata.num_rows for path in (root / "metadata" / "uid").rglob("*.parquet"))
    checks["uid_metadata_preserved"] = uid_input == uid_output
    return checks, unseen_counts


def write_reports(schema, selection, stats, rules, counters, checks, input_rows):
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "feature_columns.csv", selection, list(selection[0]))
    missing_rows = []
    for name in rules["numeric"]:
        missing_rows.append({"column": name, "kind": "numeric", "train_null": stats[name]["null"],
                             "train_dash": 0, "rule": f"fill with train minimum - 1: {rules['numeric_fill'][name]}",
                             "indicator": name + "__was_missing" in rules["feature_columns"]})
    for name in rules["categorical"]:
        missing_rows.append({"column": name, "kind": "categorical", "train_null": stats[name]["null"],
                             "train_dash": stats[name]["dash"], "rule": "'-' remains category; blank/null gets missing flag; unseen gets unknown flag",
                             "indicator": True})
    write_csv(REPORT / "missing_value_treatment.csv", missing_rows,
              ["column", "kind", "train_null", "train_dash", "rule", "indicator"])
    write_csv(REPORT / "feature_validation.csv", [{"check": key, "result": "PASS" if value else "FAIL"} for key, value in checks.items()], ["check", "result"])
    leakage = {
        "no_label_or_type_in_X": "label" not in rules["feature_columns"] and "type" not in rules["feature_columns"],
        "no_ip_time_or_trace_in_X": not set(rules["feature_columns"]) & EXCLUDED_IDENTIFIERS,
        "rules_fitted_on_train_only": checks["train_only_rules"],
        "same_features_all_splits": checks["feature_consistency"],
        "labels_and_rows_preserved": checks["label_preservation"] and checks["row_preservation"],
    }
    write_csv(REPORT / "leakage_check.csv", [{"check": key, "result": "PASS" if value else "FAIL"} for key, value in leakage.items()], ["check", "result"])
    selected = [row["column"] for row in selection if row["selected"]]
    excluded = [f"{row['column']} ({row['reason']})" for row in selection if not row["selected"]]
    lines = ["# Xử lý đặc trưng ToN-IoT Network", "",
             f"Nguồn có {len(schema.names)} cột. Chọn {len(selected)} cột gốc gồm {len(rules['numeric'])} số và {len(rules['categorical'])} phân loại; sau mã hóa có {len(rules['feature_columns'])} cột X.",
             "Cột được chọn: " + ", ".join(selected) + ".",
             "Cột không đưa vào X: " + "; ".join(excluded) + ".",
             "", "## Quy tắc", "",
             "Đọc toàn bộ train để học danh mục và giá trị nhỏ nhất của cột số. Validation/test không tham gia học quy tắc.",
             "Số thiếu được điền bằng train minimum - 1 (giá trị phân biệt với dữ liệu quan sát), có cờ thiếu cho cột thiếu trong train. Không chuẩn hóa dữ liệu số.",
             "Nếu train minimum là 1, sentinel sẽ là 0; cờ thiếu vẫn phân biệt giá trị điền với giá trị quan sát. Đây không phải quy tắc thay mọi dấu '-' bằng 0.",
             "Cột ít loại được one-hot theo danh mục train. Dấu '-' giữ thành loại riêng; blank/null và loại chưa gặp có cờ riêng. Không tạo ma trận toàn bộ dữ liệu trong RAM.",
             f"Với phần tối đa 100.000 dòng, mảng X thô ước tính khoảng {100_000 * (8 * len(rules['numeric']) + len(rules['feature_columns']) - len(rules['numeric'])) / 1_000_000:.1f} MB (chưa gồm bộ nhớ tạm của pandas/Arrow); chương trình chỉ giữ một phần trong RAM.",
             f"Nếu nạp toàn bộ X train dạng mảng vào RAM, riêng giá trị thô ước tính {input_rows['train'] * (8 * len(rules['numeric']) + len(rules['feature_columns']) - len(rules['numeric'])) / 1_000_000_000:.2f} GB; huấn luyện mô hình sẽ cần thêm bộ nhớ đáng kể.",
             "Cổng được giữ cho mô hình chính nhưng có thể mã hóa môi trường mạng; cần thử loại cổng trong phân tích độ nhạy về sau.",
             "IP, thời gian, nhãn, URI, query và định danh được giữ ở dữ liệu đã chia/metadata, không nằm trong X. Điều này giảm nguy cơ học thuộc, không chứng minh loại bỏ mọi rò rỉ gián tiếp.",
             "", "## Dữ liệu đầu ra", ""]
    for name in SPLITS:
        rows = counters[name][("__all__", "rows")]
        unseen = sum(count for (column, metric), count in counters[name].items() if metric == "unseen")
        lines.append(f"- {name}: {rows:,} dòng; normal {counters[name][('__all__', 'normal')]:,}; attack {counters[name][('__all__', 'attack')]:,}; ô phân loại chưa gặp trong train {unseen:,}.")
    lines += ["", "## Kiểm tra", "", *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()],
              "", f"Kết quả cuối: **{'PASS' if all(checks.values()) else 'FAIL'}**. Chưa huấn luyện mô hình hoặc xây dựng GNN."]
    (REPORT / "feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("========== NETWORK FEATURE PROCESSING ==========")
    print(f"Original columns: {len(schema.names)}; numeric: {len(rules['numeric'])}; categorical: {len(rules['categorical'])}; final X: {len(rules['feature_columns'])}")
    for name in SPLITS:
        print(f"{name}: {counters[name][('__all__', 'rows')]:,} rows")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(f"FINAL STATUS: {'PASS' if all(checks.values()) else 'FAIL'}")


def main():
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError("Feature output or staging exists; move it aside before rerunning")
    paths, schema, input_rows = inspect_inputs()
    stats = fit_train(paths["train"], schema)
    numeric, categorical, selection = select_features(schema, stats)
    rules = make_rules(numeric, categorical, stats, input_rows["train"])
    STAGING.mkdir(parents=True)
    rules_path = STAGING / "metadata" / "train_fitted_rules.json"
    rules_path.parent.mkdir(parents=True)
    rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    counters = {name: Counter() for name in SPLITS}
    for name in SPLITS:
        export_split(name, paths[name], rules, counters)
    checks, unseen_counts = validate_outputs(paths, rules, input_rows)
    write_csv(REPORT / "unseen_categories.csv", [
        {"split": name, "column": column, "unseen_cells": unseen_counts[(name, column)]}
        for name in SPLITS for column in rules["categorical"]
    ], ["split", "column", "unseen_cells"])
    if all(checks.values()):
        STAGING.rename(OUTPUT)
    write_reports(schema, selection, stats, rules, counters, checks, input_rows)
    return 0 if all(checks.values()) else 1


def validate_only():
    """Audit the promoted output against every source part without refitting."""
    paths, _, input_rows = inspect_inputs()
    if not OUTPUT.exists():
        raise FileNotFoundError("Official feature output is missing")
    rules = json.loads((OUTPUT / "metadata" / "train_fitted_rules.json").read_text(encoding="utf-8"))
    checks, unseen_counts = validate_outputs(paths, rules, input_rows, root=OUTPUT)
    write_csv(REPORT / "unseen_categories.csv", [
        {"split": name, "column": column, "unseen_cells": unseen_counts[(name, column)]}
        for name in SPLITS for column in rules["categorical"]
    ], ["split", "column", "unseen_cells"])
    write_csv(REPORT / "feature_validation.csv", [
        {"check": key, "result": "PASS" if value else "FAIL"} for key, value in checks.items()
    ], ["check", "result"])
    summary_path = REPORT / "feature_summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("Đối chiếu từng hàng X/y/metadata với nguồn: PASS.\n\n", "")
    summary = summary.replace("Đối chiếu từng hàng X/y/metadata với nguồn: FAIL.\n\n", "")
    summary = summary.replace("Kết quả cuối: **PASS**.",
                              "Đối chiếu từng hàng X/y/metadata với nguồn: " + ("PASS" if all(checks.values()) else "FAIL") + ".\n\nKết quả cuối: **" + ("PASS" if all(checks.values()) else "FAIL") + "**.")
    summary_path.write_text(summary, encoding="utf-8")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(f"FINAL STATUS: {'PASS' if all(checks.values()) else 'FAIL'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(validate_only() if "--validate-only" in sys.argv else main())
