"""Clean Network CSVs in chunks while retaining labels and source traceability."""

from collections import Counter
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ton_iot.config.paths import DATA_OUTPUT_DIR, DATA_PREPROCESSING_DIR, DATA_REPORT_DIR
from .reader import CHUNK_SIZE, EXPECTED_FILE_COUNT, discover_network_files
from .verifier import hashes, missing_mask


BASE = DATA_PREPROCESSING_DIR
OUTPUT = DATA_OUTPUT_DIR / "cleaned"
REPORT = DATA_REPORT_DIR / "cleaning"
VERIFICATION = DATA_REPORT_DIR / "verification"

INTEGER_COLUMNS = (
    "ts", "src_port", "dst_port", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_qclass",
    "dns_qtype", "dns_rcode", "http_trans_depth", "http_request_body_len",
    "http_response_body_len", "http_status_code",
)
FLOAT_COLUMNS = ("duration",)
NUMERIC_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS
TRACE_COLUMNS = ("record_id", "source_file", "source_row", "duplicate_key")


def source_schema(files):
    """Require the same ordered 46-column core in every source file."""
    core = list(pd.read_csv(files[0], nrows=0).columns)
    if "uid" in core:
        raise ValueError("First source file unexpectedly contains uid")
    for path in files:
        columns = list(pd.read_csv(path, nrows=0).columns)
        if [name for name in columns if name != "uid"] != core:
            raise ValueError(f"Common schema differs: {path.name}")
        if "uid" in columns and path.name != "Network_dataset_6.csv":
            raise ValueError(f"Unexpected uid column in {path.name}")
    if any(name not in core for name in NUMERIC_COLUMNS + ("label", "type", "src_ip", "proto", "service")):
        raise ValueError("Required field missing from common schema")
    return core


def read_quarantine_evidence():
    """Load verified row locations, and preserve the raw-match confidence flag."""
    suspects = {}
    with (VERIFICATION / "invalid_records.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["src_bytes_invalid"] != "True" or row["src_ip_zero"] != "True":
                raise ValueError("Verification report has a different suspect rule")
            suspects[(row["file"], int(row["csv_record_number"]))] = None
    with (VERIFICATION / "raw_source_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["file"], int(row["csv_record_number"])
            if key not in suspects:
                raise ValueError(f"Raw comparison references unknown row: {key}")
            suspects[key] = "bro_trace_found" if int(row["raw_matches"]) else "bro_trace_not_found"
    if any(value is None for value in suspects.values()):
        raise ValueError("Raw source comparison does not cover every suspect")
    return suspects


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def convert_numeric(frame, raw, file_name, first_row, stats, issues):
    """Convert numeric fields; record every nonmissing value that cannot fit."""
    for name in NUMERIC_COLUMNS:
        original = raw[name].astype("string").str.strip()
        absent = missing_mask(raw[name])
        values = pd.to_numeric(original.mask(absent), errors="coerce")
        valid = (~absent) & values.notna() & np.isfinite(values)
        if name in INTEGER_COLUMNS:
            valid &= values.mod(1).eq(0)
            # Values near int64 limits may already have lost integer precision.
            valid &= values.ge(-(2**63)) & values.lt(2**63)
            if name in ("src_port", "dst_port"):
                valid &= values.between(0, 65535)
        bad = (~absent) & ~valid
        stats["numeric_converted"][name] += int(valid.sum())
        stats["numeric_missing"][name] += int(absent.sum())
        stats["numeric_invalid"][name] += int(bad.sum())
        for offset in np.flatnonzero(bad.to_numpy()):
            issues.append({"source_file": file_name, "source_row": first_row + int(raw.index[offset]),
                           "column": name, "raw_value": original.iloc[offset],
                           "reason": "cannot_convert_without_loss"})
        # Retain raw text in the side report; never invent a numerical value.
        values = values.mask(~valid)
        frame[name] = values.astype("Int64" if name in INTEGER_COLUMNS else "Float64")
    return frame


def add_trace(frame, raw, path, first_row, core):
    rows = np.arange(first_row, first_row + len(raw), dtype=np.int64)
    first, second = hashes(raw[core])
    frame["record_id"] = [f"{path.stem}:{row}" for row in rows]
    frame["source_file"] = path.name
    frame["source_row"] = rows
    frame["duplicate_key"] = [f"{a:016x}{b:016x}" for a, b in zip(first, second)]
    return frame


def split_quarantine(frame, raw, path, first_row, suspects, stats):
    """Quarantine only verified bad-value pairs; stop if a new pair appears."""
    bad_bytes = raw["src_bytes"].str.strip().eq("0.0.0.0")
    zero_ip = raw["src_ip"].str.strip().eq("0")
    positions = np.flatnonzero((bad_bytes & zero_ip).to_numpy())
    expected = {row for file_name, row in suspects if file_name == path.name and first_row <= row < first_row + len(raw)}
    found = {first_row + int(offset) for offset in positions}
    if found != expected:
        raise ValueError(f"Quarantine evidence disagrees with source {path.name}, rows {sorted(found ^ expected)[:5]}")
    mask = np.zeros(len(raw), dtype=bool)
    mask[positions] = True
    quarantined = frame.loc[mask].copy()
    if len(quarantined):
        quarantined["quarantine_reason"] = "src_ip_zero_and_src_bytes_ip_string"
        quarantined["bro_trace_status"] = [suspects[(path.name, int(row))] for row in quarantined["source_row"]]
        # Preserve the original row, including the unconvertible src_bytes value.
        quarantined["raw_src_bytes"] = raw.loc[mask, "src_bytes"].to_numpy()
    stats["quarantine"] += len(quarantined)
    return frame.loc[~mask].copy(), quarantined


def collect_labels(frame, target):
    for (label, kind), count in frame.groupby(["label", "type"], dropna=False).size().items():
        target[(str(label), str(kind))] += int(count)


def process_file(path, core, suspects, stats, manifest):
    file_info = {"file": path.name, "input_rows": 0, "clean_rows": 0, "quarantine_rows": 0, "parts": 0}
    for part_number, raw in enumerate(pd.read_csv(
        path, chunksize=CHUNK_SIZE, dtype="string", keep_default_na=False,
        na_filter=False, on_bad_lines="error",
    ), start=1):
        first_row = file_info["input_rows"] + 2
        file_info["input_rows"] += len(raw)
        collect_labels(raw, stats["labels_before"])
        frame = raw[core].copy()
        # Keep literal '-' and other text placeholders in categorical fields.
        frame = add_trace(frame, raw, path, first_row, core)
        clean, quarantine = split_quarantine(frame, raw, path, first_row, suspects, stats)
        if "uid" in raw:
            uid = pd.DataFrame({"record_id": frame["record_id"], "uid": raw["uid"]})
            write_parquet(uid, OUTPUT / "metadata" / "uid" / path.stem / f"part_{part_number:04}.parquet")
            stats["uid_rows"] += len(uid)
        if len(quarantine):
            write_parquet(quarantine, OUTPUT / "quarantine" / path.stem / f"part_{part_number:04}.parquet")
            collect_labels(quarantine, stats["labels_quarantine"])
        if len(clean):
            clean = convert_numeric(clean, raw.loc[clean.index], path.name, first_row, stats, stats["issues"])
            write_parquet(clean, OUTPUT / "network" / path.stem / f"part_{part_number:04}.parquet")
            collect_labels(clean, stats["labels_clean"])
        file_info["clean_rows"] += len(clean)
        file_info["quarantine_rows"] += len(quarantine)
        file_info["parts"] += 1
        print(f"  {path.name}: part {part_number}, clean {len(clean):,}, quarantine {len(quarantine):,}", flush=True)
    manifest.append(file_info)


def validate_outputs(files, core, stats, manifest):
    """Read output metadata and labels, plus a source trace sample per file."""
    checks = {}
    checks["file_count"] = len(files) == EXPECTED_FILE_COUNT == len(manifest)
    checks["row_conservation"] = all(row["input_rows"] == row["clean_rows"] + row["quarantine_rows"] for row in manifest)
    expected_columns = core + list(TRACE_COLUMNS)
    read_clean = Counter()
    read_quarantine = Counter()
    labels_clean = Counter()
    labels_quarantine = Counter()
    schema_ok = trace_ok = duplicate_key_ok = True
    for source in files:
        for location, row_counts, label_counts in (
            ("network", read_clean, labels_clean),
            ("quarantine", read_quarantine, labels_quarantine),
        ):
            for part in sorted((OUTPUT / location / source.stem).glob("*.parquet")):
                parquet = pq.ParquetFile(part)
                row_counts[source.name] += parquet.metadata.num_rows
                actual_columns = parquet.schema_arrow.names
                required = expected_columns if location == "network" else expected_columns + ["quarantine_reason", "bro_trace_status", "raw_src_bytes"]
                schema_ok &= actual_columns == required
                table = pq.read_table(part, columns=["record_id", "source_row", "duplicate_key", "label", "type"])
                sample = table.slice(0, 1).to_pydict()
                if sample["record_id"]:
                    trace_ok &= sample["record_id"][0] == f"{source.stem}:{sample['source_row'][0]}"
                    duplicate_key_ok &= sample["duplicate_key"][0] is not None and len(sample["duplicate_key"][0]) == 32
                labels = table.select(["label", "type"]).to_pandas()
                collect_labels(labels, label_counts)
    checks["output_rows"] = all(
        read_clean[row["file"]] == row["clean_rows"] and read_quarantine[row["file"]] == row["quarantine_rows"]
        for row in manifest
    )
    checks["schema"] = schema_ok
    checks["trace_samples"] = trace_ok
    checks["duplicate_keys_present"] = duplicate_key_ok
    checks["labels_clean_unchanged"] = labels_clean == stats["labels_clean"]
    checks["labels_quarantine_unchanged"] = labels_quarantine == stats["labels_quarantine"]
    checks["labels_conserved"] = stats["labels_before"] == labels_clean + labels_quarantine
    checks["numeric_errors_reported"] = sum(stats["numeric_invalid"].values()) == len(stats["issues"])
    uid_parts = sorted((OUTPUT / "metadata" / "uid" / "Network_dataset_6").glob("*.parquet"))
    uid_rows = sum(pq.ParquetFile(part).metadata.num_rows for part in uid_parts)
    checks["uid_preserved"] = uid_rows == stats["uid_rows"] == next((row["input_rows"] for row in manifest if row["file"] == "Network_dataset_6.csv"), 0)
    checks["source_trace_audit"] = audit_source_samples(files)
    checks["duplicate_groups_preserved"] = audit_duplicate_groups(files, manifest)
    return checks


def audit_duplicate_groups(files, manifest):
    """Recount two-hash duplicate groups from saved clean and quarantine rows."""
    total_rows = sum(row["input_rows"] for row in manifest)
    dtype = np.dtype([("first", "u8"), ("second", "u8")])
    REPORT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="duplicate_audit_", dir=REPORT) as temp:
        keys = np.memmap(Path(temp) / "keys.bin", dtype=dtype, mode="w+", shape=(total_rows,))
        offset = 0
        for source in files:
            for location in ("network", "quarantine"):
                for part in sorted((OUTPUT / location / source.stem).glob("*.parquet")):
                    values = pq.read_table(part, columns=["duplicate_key"]).column(0).to_pylist()
                    count = len(values)
                    if any(value is None or len(value) != 32 for value in values):
                        keys._mmap.close()
                        return False
                    keys["first"][offset:offset + count] = np.fromiter((int(value[:16], 16) for value in values), dtype=np.uint64, count=count)
                    keys["second"][offset:offset + count] = np.fromiter((int(value[16:], 16) for value in values), dtype=np.uint64, count=count)
                    offset += count
        if offset != total_rows:
            keys._mmap.close()
            return False
        keys.sort(order=["first", "second"])
        groups = extra = 0
        start = 0
        while start < total_rows:
            stop = min(start + 1_000_000, total_rows)
            while stop < total_rows and keys[stop - 1] == keys[stop]:
                stop += 1
            block = keys[start:stop]
            boundary = np.flatnonzero((block["first"][1:] != block["first"][:-1]) | (block["second"][1:] != block["second"][:-1])) + 1
            lengths = np.diff(np.r_[0, boundary, len(block)])
            groups += int((lengths > 1).sum())
            extra += int((lengths[lengths > 1] - 1).sum())
            start = stop
        keys._mmap.close()
    with (VERIFICATION / "duplicate_verification.csv").open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["metric"]: int(row["count"]) for row in csv.DictReader(handle)}
    (REPORT / "duplicate_audit.json").write_text(
        json.dumps({"groups": groups, "extra_rows": extra,
                    "expected_groups": expected["groups"], "expected_extra_rows": expected["extra"]}, indent=2),
        encoding="utf-8",
    )
    return groups == expected["groups"] and extra == expected["extra"]


def audit_source_samples(files):
    """Compare one saved clean row per file and each quarantine file to raw CSV."""
    audit = []
    columns = ["ts", "src_ip", "dst_ip", "label", "type"]
    for path in files:
        targets = {}
        for location in ("network", "quarantine"):
            parts = sorted((OUTPUT / location / path.stem).glob("*.parquet"))
            if not parts:
                continue
            sample = pq.read_table(parts[0], columns=["source_row", *columns]).slice(0, 1).to_pydict()
            row_number = int(sample["source_row"][0])
            targets[row_number] = (location, {name: str(sample[name][0]) for name in columns})
        if not targets:
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            for row_number, fields in enumerate(reader, start=2):
                if row_number in targets:
                    location, output_values = targets[row_number]
                    source_values = {name: fields[header.index(name)] for name in columns}
                    audit.append({"source_file": path.name, "source_row": row_number,
                                  "output_location": location, "values_match": output_values == source_values})
                if row_number >= max(targets):
                    break
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "source_trace_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "source_row", "output_location", "values_match"])
        writer.writeheader()
        writer.writerows(audit)
    return len(audit) >= len(files) and all(row["values_match"] for row in audit)


def write_report(files, core, stats, manifest, checks):
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "file_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (REPORT / "numeric_issues.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "source_row", "column", "raw_value", "reason"])
        writer.writeheader()
        writer.writerows(stats["issues"])
    with (REPORT / "label_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "label", "type", "count"])
        for scope, counts in (("input", stats["labels_before"]), ("clean", stats["labels_clean"]), ("quarantine", stats["labels_quarantine"])):
            for (label, kind), count in sorted(counts.items()):
                writer.writerow([scope, label, kind, count])
    rules = {
        "core_columns": core, "integer_columns": INTEGER_COLUMNS, "float_columns": FLOAT_COLUMNS,
        "trace_columns": TRACE_COLUMNS, "uid_location": "metadata/uid/Network_dataset_6",
        "categorical_missing": "literal source token preserved",
        "numeric_missing": "nullable number; raw source recoverable by record_id",
        "duplicate_key": "two independently seeded pandas 64-bit hashes on original 46 columns; metadata only",
        "quarantine": "only src_ip=0 AND src_bytes=0.0.0.0, confirmed by verification report",
    }
    (REPORT / "schema_and_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    input_rows = sum(row["input_rows"] for row in manifest)
    clean_rows = sum(row["clean_rows"] for row in manifest)
    quarantine_rows = sum(row["quarantine_rows"] for row in manifest)
    lines = [
        "# Báo cáo làm sạch ToN-IoT Network", "",
        f"Tệp nguồn: {len(files)}. Dòng đầu vào: {input_rows:,}; sạch: {clean_rows:,}; cách ly: {quarantine_rows:,}; xóa vĩnh viễn: 0.",
        f"Chuyển số thành công: {sum(stats['numeric_converted'].values()):,} ô; giá trị số thiếu: {sum(stats['numeric_missing'].values()):,} ô; lỗi chuyển đổi còn lại: {sum(stats['numeric_invalid'].values()):,} ô.",
        "Cột số chuyển về Int64/Float64 có thể null; cột văn bản và nhãn giữ nguyên giá trị. Không điền 0 cho giá trị vắng.",
        "Dòng cách ly được giữ nguyên 46 cột, nhãn, vị trí, giá trị lỗi và trạng thái đối chiếu Bro; không đưa vào tập sạch.",
        "Mã truy vết và khóa nhóm trùng là metadata, không được dùng làm đặc trưng huấn luyện.",
        "Mã nhóm trùng gồm hai băm độc lập; kiểm tra bước 2 không thấy va chạm ở băm đầu, nhưng không thể loại trừ tuyệt đối va chạm kép.",
        "Không xóa dòng trùng. Nguồn gốc 148 dòng ICMP chưa ghép được về Bro vẫn chưa rõ. Các dấu '-' trong văn bản được giữ để phân biệt về sau.",
        "", "## Kiểm tra đầu ra", "",
    ]
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
    status = "PASS" if all(checks.values()) else "FAIL"
    lines += ["", f"Kết quả cuối: **{status}**. Chỉ thực hiện làm sạch; chưa chia tập hoặc huấn luyện."]
    (REPORT / "cleaning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Input {input_rows:,}; clean {clean_rows:,}; quarantine {quarantine_rows:,}; {status}")
    return status == "PASS"


def validate_only():
    """Rebuild the validation report from saved Parquet after an interrupted audit."""
    _, files, _ = discover_network_files()
    core = source_schema(files)
    analysis = BASE / "reports" / "analysis"
    input_by_file = Counter()
    labels_before = Counter()
    with (analysis / "attack_distribution.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "file":
                continue
            count = int(row["count"])
            input_by_file[row["file"]] += count
            labels_before[("0" if row["type"] == "normal" else "1", row["type"])] += count
    manifest = []
    parquet_nulls = Counter()
    labels_quarantine = Counter()
    quarantine_missing = Counter()
    for path in files:
        clean_parts = sorted((OUTPUT / "network" / path.stem).glob("*.parquet"))
        quarantine_parts = sorted((OUTPUT / "quarantine" / path.stem).glob("*.parquet"))
        if not clean_parts:
            raise FileNotFoundError(f"No clean Parquet for {path.name}")
        clean_rows = 0
        for part in clean_parts:
            parquet = pq.ParquetFile(part)
            clean_rows += parquet.metadata.num_rows
            names = parquet.schema_arrow.names
            for name in NUMERIC_COLUMNS:
                column = parquet.metadata.row_group(0).column(names.index(name))
                parquet_nulls[name] += column.statistics.null_count
        quarantine_rows = 0
        for part in quarantine_parts:
            table = pq.read_table(part, columns=["label", "type", *NUMERIC_COLUMNS]).to_pandas()
            quarantine_rows += len(table)
            collect_labels(table, labels_quarantine)
            for name in NUMERIC_COLUMNS:
                quarantine_missing[name] += int(missing_mask(table[name]).sum())
        manifest.append({"file": path.name, "input_rows": input_by_file[path.name],
                         "clean_rows": clean_rows, "quarantine_rows": quarantine_rows,
                         "parts": len(clean_parts)})
    source_missing = Counter()
    with (analysis / "missing_values.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] == "overall" and row["column"] in NUMERIC_COLUMNS:
                source_missing[row["column"]] = int(row["missing_like"])
    expected_nulls = Counter({name: source_missing[name] - quarantine_missing[name] for name in NUMERIC_COLUMNS})
    numeric_nulls_match = all(parquet_nulls[name] == expected_nulls[name] for name in NUMERIC_COLUMNS)
    labels_clean = labels_before - labels_quarantine
    uid_parts = (OUTPUT / "metadata" / "uid" / "Network_dataset_6").glob("*.parquet")
    stats = {
        "labels_before": labels_before, "labels_clean": labels_clean,
        "labels_quarantine": labels_quarantine,
        "numeric_converted": Counter({name: sum(row["clean_rows"] for row in manifest) - parquet_nulls[name] for name in NUMERIC_COLUMNS}),
        "numeric_missing": parquet_nulls, "numeric_invalid": Counter(),
        "issues": [], "uid_rows": sum(pq.ParquetFile(part).metadata.num_rows for part in uid_parts),
    }
    checks = validate_outputs(files, core, stats, manifest)
    checks["numeric_nulls_match_source"] = numeric_nulls_match
    return 0 if write_report(files, core, stats, manifest, checks) else 1


def main():
    _, files, _ = discover_network_files()
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FILE_COUNT} files; found {len(files)}")
    core = source_schema(files)
    suspects = read_quarantine_evidence()
    if OUTPUT.exists():
        raise FileExistsError(f"Output directory exists: {OUTPUT}. Move it aside before a new run.")
    OUTPUT.mkdir(parents=True)
    stats = {
        "labels_before": Counter(), "labels_clean": Counter(), "labels_quarantine": Counter(),
        "numeric_converted": Counter(), "numeric_missing": Counter(), "numeric_invalid": Counter(),
        "issues": [], "quarantine": 0, "uid_rows": 0,
    }
    manifest = []
    for path in files:
        print(f"Cleaning {path.name} ...", flush=True)
        process_file(path, core, suspects, stats, manifest)
    checks = validate_outputs(files, core, stats, manifest)
    return 0 if write_report(files, core, stats, manifest, checks) else 1


if __name__ == "__main__":
    if "--validate-only" in sys.argv:
        raise SystemExit(validate_only())
    if "--audit-only" in sys.argv:
        _, source_files, _ = discover_network_files()
        raise SystemExit(0 if audit_source_samples(source_files) else 1)
    raise SystemExit(main())
