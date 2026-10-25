"""Verify unresolved Network dataset issues without changing source CSV files."""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from ton_iot.config.paths import DATA_REPORT_DIR, PROJECT_ROOT, RAW_DATA_DIR
from .analyzer import CONNECTION_COLUMNS, MISSING_NULL_TOKENS, MISSING_PLACEHOLDER_TOKENS
from .reader import CHUNK_SIZE, EXPECTED_FILE_COUNT, discover_network_files


OUTPUT = DATA_REPORT_DIR / "verification"
ANALYSIS = DATA_REPORT_DIR / "analysis"
HASH_KEY_2 = "0123456789abcdef"
RECORD_DTYPE = np.dtype([("key", "u8"), ("check", "u8"), ("file", "u1"), ("kind", "u1")])
CONNECTION_DTYPE = np.dtype([("key", "u8"), ("check", "u8")])


def write_csv(name, rows, columns):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def hashes(frame):
    """Two independently seeded 64-bit hashes over identically ordered values."""
    first = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    second = pd.util.hash_pandas_object(frame, index=False, hash_key=HASH_KEY_2).to_numpy(dtype=np.uint64)
    return first, second


def missing_mask(series):
    values = series.astype("string").str.strip().str.lower()
    return values.isna() | values.eq("") | values.isin(MISSING_NULL_TOKENS | MISSING_PLACEHOLDER_TOKENS)


def check_schema(files):
    reference = list(pd.read_csv(files[0], nrows=0).columns)
    rows = []
    for path in files:
        columns = list(pd.read_csv(path, nrows=0).columns)
        shared = [column for column in columns if column in reference]
        sample = pd.read_csv(path, nrows=CHUNK_SIZE, low_memory=False)
        rows.append({
            "file": path.name,
            "column_count": len(columns),
            "shared_columns_same_order": shared == reference,
            "extra_columns": ";".join(column for column in columns if column not in reference),
            "missing_columns": ";".join(column for column in reference if column not in columns),
            "src_bytes_sample_dtype": str(sample["src_bytes"].dtype),
            "duration_sample_dtype": str(sample["duration"].dtype),
        })
    write_csv("schema_verification.csv", rows, list(rows[0]))
    if any(row["missing_columns"] or not row["shared_columns_same_order"] for row in rows):
        raise ValueError("Shared columns differ across files; canonical duplicate key is undefined")
    return reference, rows


def inspect_file(path, file_number, shared, kinds, record_path, connection_path, state):
    for chunk_number, chunk in enumerate(pd.read_csv(
        path, chunksize=CHUNK_SIZE, dtype="string", keep_default_na=False,
        na_filter=False, on_bad_lines="error",
    )):
        first_row = chunk_number * CHUNK_SIZE + 2
        state["rows"] += len(chunk)
        state["file_rows"][path.name] += len(chunk)
        src_invalid = (~missing_mask(chunk["src_bytes"])) & pd.to_numeric(chunk["src_bytes"], errors="coerce").isna()
        ip_zero = chunk["src_ip"].str.strip().eq("0")
        state["invalid_src_bytes"] += int(src_invalid.sum())
        state["zero_src_ip"] += int(ip_zero.sum())
        state["invalid_overlap"] += int((src_invalid & ip_zero).sum())
        for offset in np.flatnonzero((src_invalid | ip_zero).to_numpy()):
            row = chunk.iloc[int(offset)]
            state["invalid_rows"].append({
                "file": path.name, "csv_record_number": first_row + int(offset),
                "src_bytes_invalid": bool(src_invalid.iloc[offset]),
                "src_ip_zero": bool(ip_zero.iloc[offset]),
                "parsed_column_count": len(row),
                "parsed_values": "|".join(f"{key}={value}" for key, value in row.items()),
                "_parsed_fields": row.tolist(),
            })

        key, check = hashes(chunk[shared])
        records = np.empty(len(chunk), dtype=RECORD_DTYPE)
        records["key"], records["check"] = key, check
        records["file"] = file_number
        records["kind"] = chunk["type"].str.strip().str.lower().map(kinds).to_numpy(dtype=np.uint8)
        with record_path.open("ab") as handle:
            records.tofile(handle)

        columns = ["ts", *CONNECTION_COLUMNS]
        first, second = hashes(chunk[columns])
        connections = np.empty(len(chunk), dtype=CONNECTION_DTYPE)
        connections["key"], connections["check"] = first, second
        with connection_path.open("ab") as handle:
            connections.tofile(handle)

        groups = {}
        for name in ("proto", "service"):
            codes, labels = pd.factorize(chunk[name].str.strip().str.lower(), sort=False)
            groups[name] = (codes, labels, np.bincount(codes, minlength=len(labels)))
        for column in chunk.columns:
            if not column.startswith(("http_", "ssl_", "dns_")) and column != "service":
                continue
            missing = missing_mask(chunk[column]).to_numpy(dtype=np.int8)
            for group_name, (codes, labels, totals) in groups.items():
                absent = np.bincount(codes, weights=missing, minlength=len(labels))
                for group_value, total, missing_count in zip(labels, totals, absent):
                    bucket = state["missing"][(column, group_name, str(group_value))]
                    bucket["present"] += int(total - missing_count)
                    bucket["missing_like"] += int(missing_count)


def audit_raw_records(files, rows):
    """Use csv.reader line_num to check physical lines and parsed field alignment."""
    by_file = defaultdict(dict)
    for row in rows:
        by_file[row["file"]][row["csv_record_number"]] = row
    for path in files:
        targets = by_file.get(path.name)
        if not targets:
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            for record_number, fields in enumerate(reader, start=2):
                if record_number in targets:
                    target = targets[record_number]
                    target["physical_line_end"] = reader.line_num
                    target["raw_field_count"] = len(fields)
                    target["field_count_matches_header"] = len(fields) == len(header)
                    target["raw_src_ip"] = fields[header.index("src_ip")] if len(fields) == len(header) else ""
                    target["raw_src_bytes"] = fields[header.index("src_bytes")] if len(fields) == len(header) else ""
                    target["raw_matches_parsed"] = (
                        len(fields) == len(header)
                        and fields == target["_parsed_fields"]
                    )
                    del target["_parsed_fields"]
                if record_number >= max(targets):
                    break


def summarize_duplicates(path, dtype, kinds=None):
    """Sort disk-backed keys, then count groups in bounded slices."""
    count = path.stat().st_size // dtype.itemsize
    values = np.memmap(path, dtype=dtype, mode="r+", shape=(count,))
    values.sort(order=["key", "check"])
    result = Counter()
    by_type = Counter()
    by_file = Counter()
    start = 0
    while start < count:
        stop = min(start + 500_000, count)
        while stop < count and values[stop - 1]["key"] == values[stop]["key"] and values[stop - 1]["check"] == values[stop]["check"]:
            stop += 1
        block = values[start:stop]
        first = block["key"]
        second = block["check"]
        boundary = np.flatnonzero((first[1:] != first[:-1]) | (second[1:] != second[:-1])) + 1
        starts = np.r_[0, boundary]
        ends = np.r_[boundary, len(block)]
        duplicated = ends - starts > 1
        result["groups"] += int(duplicated.sum())
        result["extra"] += int((ends[duplicated] - starts[duplicated] - 1).sum())
        first_only = np.flatnonzero(first[1:] != first[:-1]) + 1
        result["first_hash_collision_groups"] += int(len(boundary) - len(first_only))
        if kinds is not None and duplicated.any():
            duplicate_starts = starts[duplicated]
            duplicate_ends = ends[duplicated]
            lengths = duplicate_ends - duplicate_starts - 1
            result["mixed_type_groups"] += int(np.sum(
                np.minimum.reduceat(block["kind"], starts)[duplicated]
                != np.maximum.reduceat(block["kind"], starts)[duplicated]
            ))
            result["cross_file_groups"] += int(np.sum(
                np.minimum.reduceat(block["file"], starts)[duplicated]
                != np.maximum.reduceat(block["file"], starts)[duplicated]
            ))
            for kind, extra in zip(block["kind"][duplicate_starts], lengths):
                by_type[int(kind)] += int(extra)
            for left, right in zip(duplicate_starts, duplicate_ends):
                files, counts = np.unique(block["file"][left:right], return_counts=True)
                for file_id, file_count in zip(files, counts):
                    by_file[int(file_id)] += int(file_count - 1)
                if len(files) > 1:
                    result["cross_file_extra"] += int(len(files) - 1)
                    result["cross_file_group_extra"] += int(right - left - 1)
        start = stop
    values._mmap.close()
    return result, by_type, by_file


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, files, warnings = discover_network_files()
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FILE_COUNT} files, found {len(files)}")
    if "--summary-only" in sys.argv:
        return summary_only(files, warnings)
    shared, schemas = check_schema(files)
    attack_types = set()
    for path in files:
        for part in pd.read_csv(path, usecols=["type"], dtype="string", chunksize=CHUNK_SIZE):
            attack_types.update(part["type"].str.strip().str.lower().dropna().unique().tolist())
    attack_types = sorted(attack_types)
    kinds = {name: number for number, name in enumerate(attack_types)}
    state = {
        "rows": 0, "file_rows": Counter(), "invalid_rows": [],
        "invalid_src_bytes": 0, "zero_src_ip": 0, "invalid_overlap": 0,
        "missing": defaultdict(Counter),
    }
    with tempfile.TemporaryDirectory(prefix="verify_", dir=OUTPUT) as temp:
        record_path = Path(temp) / "records.bin"
        connection_path = Path(temp) / "connections.bin"
        for number, path in enumerate(files):
            print(f"Verifying {path.name} ...", flush=True)
            inspect_file(path, number, shared, kinds, record_path, connection_path, state)
        audit_raw_records(files, state["invalid_rows"])
        write_csv("invalid_records.csv", state["invalid_rows"], [
            "file", "csv_record_number", "physical_line_end", "src_bytes_invalid", "src_ip_zero",
            "parsed_column_count", "raw_field_count", "field_count_matches_header", "raw_matches_parsed",
            "raw_src_ip", "raw_src_bytes", "parsed_values",
        ])
        compare_raw_sources()
        exact, by_type, by_file = summarize_duplicates(record_path, RECORD_DTYPE, kinds)
        same_connection, _, _ = summarize_duplicates(connection_path, CONNECTION_DTYPE)
    duplicate_rows = [
        {"metric": key, "count": count} for key, count in sorted(exact.items())
    ] + [{"metric": "same_timestamp_and_connection_" + key, "count": count} for key, count in sorted(same_connection.items())]
    duplicate_rows += [{"metric": "within_file_extra:" + files[file_id].name, "count": count} for file_id, count in sorted(by_file.items())]
    write_csv("duplicate_verification.csv", duplicate_rows, ["metric", "count"])
    type_rows = [{"type": name, "label": "normal" if name == "normal" else "attack", "extra_exact_rows": by_type[kinds[name]]} for name in attack_types]
    write_csv("duplicate_by_attack_type.csv", type_rows, list(type_rows[0]))
    missing_rows = []
    for (column, group_name, group_value), counts in sorted(state["missing"].items()):
        missing_rows.append({"column": column, "group_by": group_name, "group_value": group_value,
                             "present": counts["present"], "missing_like": counts["missing_like"],
                             "rows": sum(counts.values())})
    write_csv("missing_value_verification.csv", missing_rows, list(missing_rows[0]))
    write_summary(files, warnings, shared, schemas, state, exact, same_connection)
    print(f"Rows: {state['rows']:,}; exact duplicate extra rows: {exact['extra']:,}; invalid src_bytes: {state['invalid_src_bytes']:,}")


def summary_only(files, warnings):
    """Regenerate the narrative after an interrupted final write, using saved evidence."""
    with (OUTPUT / "schema_verification.csv").open(encoding="utf-8-sig") as handle:
        schemas = list(csv.DictReader(handle))
    with (OUTPUT / "invalid_records.csv").open(encoding="utf-8-sig") as handle:
        invalid_rows = list(csv.DictReader(handle))
    with (OUTPUT / "duplicate_verification.csv").open(encoding="utf-8-sig") as handle:
        metrics = {row["metric"]: int(row["count"]) for row in csv.DictReader(handle)}
    with (OUTPUT / "missing_value_verification.csv").open(encoding="utf-8-sig") as handle:
        rows = sum(int(row["rows"]) for row in csv.DictReader(handle)
                   if row["column"] == "service" and row["group_by"] == "proto")
    state = {
        "rows": rows, "invalid_rows": invalid_rows,
        "invalid_src_bytes": sum(row["src_bytes_invalid"] == "True" for row in invalid_rows),
        "zero_src_ip": sum(row["src_ip_zero"] == "True" for row in invalid_rows),
        "invalid_overlap": sum(row["src_bytes_invalid"] == "True" and row["src_ip_zero"] == "True" for row in invalid_rows),
    }
    exact = {key: value for key, value in metrics.items() if not key.startswith(("same_timestamp_and_connection_", "within_file_extra:"))}
    same_connection = {key.removeprefix("same_timestamp_and_connection_"): value for key, value in metrics.items() if key.startswith("same_timestamp_and_connection_")}
    shared = list(pd.read_csv(files[0], nrows=0).columns)
    write_summary(files, warnings, shared, schemas, state, exact, same_connection)
    print(f"Summary regenerated from saved verification CSVs: {rows:,} rows")


def compare_raw_sources():
    """Match suspect processed ICMP rows to Bro conn records by shared fields."""
    with (OUTPUT / "invalid_records.csv").open(encoding="utf-8-sig") as handle:
        suspects = list(csv.DictReader(handle))
    by_key = defaultdict(list)
    keys = ("ts", "src_port", "dst_ip", "dst_port", "proto", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes")
    for row in suspects:
        values = dict(part.split("=", 1) for part in row["parsed_values"].split("|") if "=" in part)
        by_key[tuple(values[name] for name in keys)].append(row)
        row.update({"raw_matches": 0, "raw_src_ips": set(), "raw_src_bytes_values": set(), "raw_examples": []})
    root = RAW_DATA_DIR / "network_data" / "Network_dataset_Bro"
    for path in sorted(root.rglob("conn.csv")):
        print(f"Checking raw {path.parent.name}/conn.csv ...", flush=True)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for fields in reader:
                if len(fields) < 20:
                    continue
                key = tuple(fields[index] for index in (0, 3, 4, 5, 6, 16, 17, 18, 19))
                for target in by_key.get(key, ()):
                    target["raw_matches"] += 1
                    target["raw_src_ips"].add(fields[2])
                    target["raw_src_bytes_values"].add(fields[9])
                    if len(target["raw_examples"]) < 3:
                        target["raw_examples"].append(f"{path.relative_to(root)}:{reader.line_num}")
    rows = [{
        "file": row["file"], "csv_record_number": row["csv_record_number"],
        "raw_matches": row["raw_matches"],
        "raw_src_ips": ";".join(sorted(row["raw_src_ips"])),
        "raw_src_bytes_values": ";".join(sorted(row["raw_src_bytes_values"])),
        "raw_examples": ";".join(row["raw_examples"]),
    } for row in suspects]
    write_csv("raw_source_comparison.csv", rows, list(rows[0]))
    print(f"Raw comparison: {sum(row['raw_matches'] > 0 for row in rows):,}/{len(rows):,} suspect rows matched")


def write_summary(files, warnings, shared, schemas, state, exact, same_connection):
    with (ANALYSIS / "duplicate_analysis.csv").open(encoding="utf-8-sig") as handle:
        old = {(row["scope"], row["metric"]): int(row["count"]) for row in csv.DictReader(handle)}
    old_extra = old[("all_files", "exact_duplicate_extra_rows")]
    raw_comparison = OUTPUT / "raw_source_comparison.csv"
    if raw_comparison.exists():
        with raw_comparison.open(encoding="utf-8-sig") as handle:
            raw_rows = list(csv.DictReader(handle))
        raw_matched = sum(int(row["raw_matches"]) > 0 for row in raw_rows)
        raw_ips = sorted({ip for row in raw_rows for ip in row["raw_src_ips"].split(";") if ip})
        raw_bytes = sorted({value for row in raw_rows for value in row["raw_src_bytes_values"].split(";") if value})
        unmatched_by_file = Counter(row["file"] for row in raw_rows if int(row["raw_matches"]) == 0)
        raw_evidence = f"Đối chiếu Bro `conn.csv`: {raw_matched:,}/{len(raw_rows):,} dòng tìm thấy cùng dấu vết kết nối; IP nguồn trong Bro: {raw_ips}; orig_bytes trong Bro: {raw_bytes}. Chưa tìm thấy {len(raw_rows) - raw_matched:,} dòng, theo tệp: {dict(unmatched_by_file)}. Một số bản ghi Bro có cùng dấu vết nên phép ghép không luôn duy nhất. Xem `raw_source_comparison.csv` để biết tệp/dòng nguồn."
    else:
        raw_evidence = "Chưa đối chiếu tự động với Bro conn.csv."
    with (ANALYSIS / "label_distribution_by_file.csv").open(encoding="utf-8-sig") as handle:
        previous_rows = sum(int(row["count"]) for row in csv.DictReader(handle))
    with (ANALYSIS / "missing_values.csv").open(encoding="utf-8-sig") as handle:
        old_missing = list(csv.DictReader(handle))
    overall_missing = [row for row in old_missing if row["scope"] == "overall"]
    blank_total = sum(int(row["blank"]) for row in overall_missing)
    null_total = sum(int(row["null_token"]) for row in overall_missing)
    placeholder_total = sum(int(row["placeholder"]) for row in overall_missing)
    with (OUTPUT / "missing_value_verification.csv").open(encoding="utf-8-sig") as handle:
        missing_details = list(csv.DictReader(handle))
    service_lookup = {(row["column"], row["group_value"]): row for row in missing_details if row["group_by"] == "service"}
    family_rows = []
    for column in sorted({row["column"] for row in missing_details}):
        family = column.split("_", 1)[0]
        if family not in {"http", "ssl", "dns"}:
            continue
        matched_service = service_lookup.get((column, family))
        if matched_service:
            family_rows.append(
                f"| `{column}` | {int(matched_service['present']):,} | {int(matched_service['missing_like']):,} | Trong `service={family}`: chưa rõ vì sao vắng; ngoài dịch vụ: thường không áp dụng |"
            )
    lines = [
        "# Xác minh các bất thường trong Network dataset", "",
        f"Đã đọc {len(files)} tệp, {state['rows']:,} dòng. Nguồn dữ liệu không thay đổi.", "",
        "## 1. Giá trị bất thường", "",
        f"`src_bytes` không phải số: {state['invalid_src_bytes']:,}; `src_ip = 0`: {state['zero_src_ip']:,}; giao của hai nhóm: {state['invalid_overlap']:,}.",
        "Vị trí và toàn bộ giá trị đã phân tích nằm trong `invalid_records.csv`; số trường thô được đối chiếu với header bằng csv.reader.",
        "Các dòng có số trường khớp header và giá trị thô khớp dữ liệu pandas; vì vậy không phải lỗi pandas đọc lệch cột.",
        raw_evidence,
        "Ở các dòng ghép được, Bro ghi IP nguồn `::` trong khi CSV đã xử lý ghi `0`; Bro ghi `orig_bytes` là số hoặc `-` trong khi CSV đã xử lý ghi `0.0.0.0` ở `src_bytes`. Có bằng chứng biến đổi sai trong quá trình tạo CSV đã xử lý, nhưng chưa xác định được phép biến đổi cụ thể.",
        "Tệp đã xử lý không giữ `uid` cho các dòng này, nên phép nối về Bro bằng nhiều trường kết nối có thể không xác định duy nhất bản ghi gốc. Chưa xác định được thao tác cụ thể đã tạo ra giá trị sai.",
        "Đề xuất cách ly để kiểm tra nguồn gốc; không tự điền `src_bytes=0` hoặc sửa IP.", "",
        "## 2. Bản ghi trùng", "",
        f"Báo cáo cũ: {old_extra:,} dòng bổ sung; kiểm tra trên {len(shared)} cột chung: {exact['extra']:,} dòng bổ sung thuộc {exact['groups']:,} nhóm.",
        f"Trong nhiều tệp: {exact['cross_file_groups']:,} nhóm, {exact['cross_file_group_extra']:,} dòng dư thuộc các nhóm này; trong đó {exact['cross_file_extra']:,} dòng dư phát sinh riêng do gộp tệp.",
        f"Cùng thời gian và sáu trường kết nối: {same_connection['groups']:,} nhóm, {same_connection['extra']:,} dòng dư; đây không đồng nghĩa trùng toàn dòng.",
        "Chỉ số này khác 'cùng sáu trường kết nối nhưng khác thời gian' trong báo cáo cũ; hai phép đếm có điều kiện nhóm khác nhau và không được đối chiếu như cùng một đại lượng.",
        f"Nhóm có cùng băm thứ nhất nhưng băm thứ hai khác: {exact['first_hash_collision_groups']:,}. Hai băm 64 bit độc lập giảm rủi ro, nhưng không chứng minh toán học rằng va chạm kép là bất khả thi.",
        "Tệp 6 có thêm `uid`; so sánh cũ gồm cột này, còn lần này dùng cùng bộ cột chung cho cả 23 tệp. Không xóa trùng tự động vì các bản ghi kết nối lặp có thể là sự kiện thật.",
        "Số dư theo loại tấn công nằm trong `duplicate_by_attack_type.csv`.", "",
        "## 3. Giá trị thiếu", "",
        f"Báo cáo cũ gộp `-`, rỗng và null-token thành missing-like: dấu thay thế {placeholder_total:,}, rỗng {blank_total:,}, null-token {null_total:,}. Bảng mới đếm từng cột theo `proto` và `service`.",
        "Các trường HTTP/SSL/DNS có `-` ngoài dịch vụ tương ứng thường là không áp dụng; ngay trong dịch vụ tương ứng vẫn có thể là chưa ghi nhận. `service = -` là dịch vụ không xác định, không phải giá trị số 0.",
        "Giá trị `0` trong các cột số không được chương trình cũ tính là thiếu; hiện chưa đủ bằng chứng để kết luận mọi số 0 là giá trị đo thật hay giá trị mặc định.",
        "| Cột | Có giá trị trong service cùng tên | Vắng trong service cùng tên | Mức xác nhận |",
        "| --- | ---: | ---: | --- |",
        *family_rows,
        "Với từng cột, chỉ có thể xác nhận tình trạng vắng mặt, chưa thể xác nhận nguyên nhân của từng ô vắng trong đúng dịch vụ. Các trường có giá trị ngoài dịch vụ cùng tên cần xem riêng trong CSV.",
        "Đề xuất giữ nguyên dấu vắng mặt và thêm cờ hiện diện khi thiết kế đặc trưng; chưa xóa cột theo tỷ lệ thiếu.", "",
        "## 4. Cấu trúc và kiểu dữ liệu", "",
        f"Các cột chung giữ đúng thứ tự ở {sum(str(row['shared_columns_same_order']) == 'True' for row in schemas)}/{len(schemas)} tệp; tệp 6 có thêm `uid`.",
        "`src_bytes` chứa chuỗi IP làm kiểu pandas suy luận thành object ở tệp liên quan. `duration` có thể được suy luận int/float tùy giá trị của chunk; đọc chuỗi rồi chuyển số sau khi kiểm tra sẽ ổn định.",
        "Đề xuất schema chung giữ `ts`, hai IP, hai port, `proto`, `service`, `label`, `type` và các đặc trưng khác; giữ `uid` dưới dạng cột tùy chọn (thiếu ở 22 tệp), phục vụ kiểm tra/GNN.",
        "", "## Kết luận", "",
        f"Đối chiếu số dòng bước 1: {'KHỚP' if state['rows'] == previous_rows else 'KHÁC'} ({previous_rows:,} dòng trong báo cáo cũ).",
        f"Thay đổi số liệu trùng so với báo cáo cũ: {exact['extra'] - old_extra:+,} dòng; xem nguyên nhân schema `uid` trước khi dùng ở bước 3.",
        f"Bước 2 có thể kết thúc với nguồn gốc {len(state['invalid_rows']):,} dòng bất thường được đánh dấu chưa xác minh hoàn toàn; chưa làm sạch dữ liệu.",
    ]
    if warnings:
        lines.extend(["", "Cảnh báo nguồn: " + "; ".join(warnings)])
    (OUTPUT / "verification_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--raw-compare" in sys.argv:
        compare_raw_sources()
    else:
        main()
