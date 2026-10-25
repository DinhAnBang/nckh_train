"""Time-ordered, leakage-audited split of the cleaned ToN-IoT Network data."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cleaner import BASE, OUTPUT as CLEANED, write_parquet
from .verifier import hashes


NETWORK = CLEANED / "network"
QUARANTINE = CLEANED / "quarantine"
UID = CLEANED / "metadata" / "uid"
REPORT = BASE / "reports" / "splitting"
STAGING = BASE / "outputs" / "split_staging"
FINAL = BASE / "outputs" / "split"
NAMES = ("train", "validation", "test")
TRAIN_TYPES = {"scanning", "dos", "injection", "ddos"}
VALIDATION_TYPES = {"password"}
TEST_TYPES = {"xss", "ransomware", "backdoor", "mitm"}
PROTOCOL = (TRAIN_TYPES, VALIDATION_TYPES, TEST_TYPES)
CONNECTION = ["src_ip", "src_port", "dst_ip", "dst_port", "proto", "service"]
KEY_DTYPE = np.dtype([("first", "u8"), ("second", "u8"), ("split", "u1")])


def utc(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat()


def attack_time_context(types):
    """Describe gaps and overlapping attack intervals from observed timestamps."""
    attacks = sorted(((kind, item) for kind, item in types.items() if kind != "normal"),
                     key=lambda pair: pair[1]["first"])
    gaps = []
    overlaps = []
    for (earlier_name, earlier), (later_name, later) in zip(attacks, attacks[1:]):
        difference = later["first"] - earlier["last"]
        if difference > 0:
            gaps.append(f"{earlier_name} → {later_name}: {difference:,} giây")
        else:
            overlaps.append(f"{earlier_name} / {later_name}")
    return [
        "Khoảng giữa các đợt tấn công theo thời điểm đầu/cuối: " + "; ".join(gaps) + ".",
        "Các cặp đợt tấn công liền kề có khoảng thời gian chồng lấn: " + (", ".join(overlaps) if overlaps else "không có") + ".",
    ]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def source_parts():
    """Accept only the validated clean data and return its real Parquet parts."""
    summary = (BASE / "reports" / "cleaning" / "cleaning_summary.md").read_text(encoding="utf-8")
    if "Kết quả cuối: **PASS**" not in summary:
        raise ValueError("Cleaning report is not PASS")
    parts = sorted(NETWORK.rglob("*.parquet"))
    if not parts or any("cleaned_failed" in str(path) for path in parts):
        raise ValueError("No valid clean source parts")
    with (BASE / "reports" / "cleaning" / "file_counts.csv").open(encoding="utf-8-sig") as handle:
        expected = {row["file"].removesuffix(".csv"): int(row["clean_rows"]) for row in csv.DictReader(handle)}
    actual = Counter()
    reference = pq.ParquetFile(parts[0]).schema_arrow
    for path in parts:
        parquet = pq.ParquetFile(path)
        actual[path.parent.name] += parquet.metadata.num_rows
        if parquet.schema_arrow.names != reference.names:
            raise ValueError(f"Schema differs in {path}")
    if actual != expected:
        raise ValueError(f"Clean Parquet row counts differ from cleaning report: {actual - expected}")
    required = {"ts", "label", "type", "record_id", "source_file", "source_row", "duplicate_key", *CONNECTION}
    if not required.issubset(reference.names):
        raise ValueError("Required Network fields are missing")
    return parts, reference, sum(actual.values())


def profile_time(parts):
    """Read all timestamps and labels; retain only time/type aggregates."""
    types = defaultdict(lambda: {"count": 0, "first": None, "last": None})
    daily = Counter()
    label_daily = Counter()
    for number, part in enumerate(parts, start=1):
        frame = pq.read_table(part, columns=["ts", "label", "type"]).to_pandas()
        if frame["ts"].isna().any() or frame["label"].isna().any() or frame["type"].isna().any():
            raise ValueError(f"Missing timestamp or label in {part}")
        day = frame["ts"].to_numpy(dtype=np.int64) // 86_400
        for kind, subset in frame.groupby("type"):
            item = types[str(kind)]
            item["count"] += len(subset)
            earliest, latest = int(subset["ts"].min()), int(subset["ts"].max())
            item["first"] = earliest if item["first"] is None else min(item["first"], earliest)
            item["last"] = latest if item["last"] is None else max(item["last"], latest)
        for (day_id, label, kind), count in pd.DataFrame({"day": day, "label": frame["label"], "type": frame["type"]}).groupby(["day", "label", "type"]).size().items():
            daily[(int(day_id), str(kind))] += int(count)
            label_daily[(int(day_id), str(label))] += int(count)
        if number % 25 == 0:
            print(f"Profiling time: {number}/{len(parts)} parts", flush=True)
    return types, daily, label_daily


def choose_boundaries(types):
    """Check the requested unseen-attack protocol before any official export."""
    actual_attack_types = set(types) - {"normal"}
    expected_attack_types = set().union(*PROTOCOL)
    reasons = []
    if actual_attack_types != expected_attack_types:
        reasons.append(f"Attack types differ: actual={sorted(actual_attack_types)}, requested={sorted(expected_attack_types)}")
    if any(kind not in types for kind in expected_attack_types):
        return None, reasons
    train_end = max(types[kind]["last"] for kind in TRAIN_TYPES)
    validation_end = max(types[kind]["last"] for kind in VALIDATION_TYPES)
    validation_start = min(types[kind]["first"] for kind in VALIDATION_TYPES)
    test_start = min(types[kind]["first"] for kind in TEST_TYPES)
    if not train_end < validation_start:
        reasons.append("Train attacks overlap the validation attack interval")
    if not validation_end < test_start:
        reasons.append("Validation attacks overlap the test attack interval")
    for kind in TRAIN_TYPES:
        if types[kind]["first"] > train_end:
            reasons.append(f"Train attack absent before boundary: {kind}")
    for kind in TEST_TYPES:
        if types[kind]["first"] <= validation_end:
            reasons.append(f"Test attack appears before test boundary: {kind}")
    boundary_path = REPORT / "split_boundaries.csv"
    if boundary_path.exists():
        with boundary_path.open(encoding="utf-8-sig") as handle:
            fixed = {row["boundary"]: int(row["ts"]) for row in csv.DictReader(handle)}
        if fixed.get("train_end") != train_end or fixed.get("validation_end") != validation_end:
            reasons.append("Saved boundaries differ from the current data; refusing to change the protocol silently")
        train_end, validation_end = fixed["train_end"], fixed["validation_end"]
    else:
        write_csv(boundary_path, [
            {"boundary": "train_end", "ts": train_end, "utc": utc(train_end), "reason": "last train attack timestamp"},
            {"boundary": "validation_end", "ts": validation_end, "utc": utc(validation_end), "reason": "last validation attack timestamp"},
        ], ["boundary", "ts", "utc", "reason"])
    return (train_end, validation_end), reasons


def quarantine_ids():
    ids = set()
    for path in QUARANTINE.rglob("*.parquet"):
        ids.update(pq.read_table(path, columns=["record_id"]).column(0).to_pylist())
    return ids


def record_key_batch(path, handles, boundaries, stats, near, forbidden, previous_rows):
    """Count stage labels, write temporary duplicate keys, and inspect boundaries."""
    columns = ["ts", "label", "type", "duplicate_key", "record_id", "source_row", *CONNECTION]
    frame = pq.read_table(path, columns=columns).to_pandas()
    times = frame["ts"].to_numpy(dtype=np.int64)
    stage = np.searchsorted(boundaries, times, side="left").astype(np.uint8)
    if frame["record_id"].isin(forbidden).any():
        stats["quarantine_overlap"] += int(frame["record_id"].isin(forbidden).sum())
    source = path.parent.name
    row_numbers = frame["source_row"].to_numpy(dtype=np.int64)
    if len(row_numbers) and (row_numbers[0] <= previous_rows.get(source, 0) or np.any(np.diff(row_numbers) <= 0)):
        stats["record_order_errors"] += 1
    if len(row_numbers):
        previous_rows[source] = int(row_numbers[-1])
    first_id = frame["record_id"].iloc[0]
    last_id = frame["record_id"].iloc[-1]
    if first_id != f"{source}:{row_numbers[0]}" or last_id != f"{source}:{row_numbers[-1]}":
        stats["record_id_errors"] += 1
    keys = frame["duplicate_key"].tolist()
    if any(not isinstance(value, str) or len(value) != 32 for value in keys):
        raise ValueError(f"Invalid duplicate key in {path}")
    packed = np.empty(len(keys), dtype=KEY_DTYPE)
    packed["first"] = np.fromiter((int(value[:16], 16) for value in keys), dtype=np.uint64, count=len(keys))
    packed["second"] = np.fromiter((int(value[16:], 16) for value in keys), dtype=np.uint64, count=len(keys))
    packed["split"] = stage
    packed.tofile(handles["keys"])
    for index, name in enumerate(NAMES):
        mask = stage == index
        if not mask.any():
            continue
        subset = frame.loc[mask]
        stats["rows"][name] += len(subset)
        stats["min_ts"][name] = min(stats["min_ts"].get(name, int(subset["ts"].min())), int(subset["ts"].min()))
        stats["max_ts"][name] = max(stats["max_ts"].get(name, int(subset["ts"].max())), int(subset["ts"].max()))
        for (label, kind), count in subset.groupby(["label", "type"]).size().items():
            stats["labels"][(name, str(label), str(kind))] += int(count)
    for boundary_number, boundary in enumerate(boundaries):
        close = np.abs(times - boundary) <= 3600
        if not close.any():
            continue
        left, right = hashes(frame.loc[close, CONNECTION])
        near_side = np.where(times[close] <= boundary, "before", "after")
        for a, b, side in zip(left, right, near_side):
            near[(boundary_number, side)].add((int(a), int(b)))


def audit_duplicate_keys(path, count):
    """Sort all two-hash keys to count groups and detect cross-stage copies."""
    values = np.memmap(path, dtype=KEY_DTYPE, mode="r+", shape=(count,))
    values.sort(order=["first", "second"])
    per_split = {name: Counter() for name in NAMES}
    cross_groups = cross_rows = 0
    start = 0
    while start < count:
        stop = min(start + 1_000_000, count)
        while stop < count and values[stop - 1]["first"] == values[stop]["first"] and values[stop - 1]["second"] == values[stop]["second"]:
            stop += 1
        block = values[start:stop]
        boundary = np.flatnonzero((block["first"][1:] != block["first"][:-1]) | (block["second"][1:] != block["second"][:-1])) + 1
        starts, ends = np.r_[0, boundary], np.r_[boundary, len(block)]
        repeated = ends - starts > 1
        for left, right in zip(starts[repeated], ends[repeated]):
            length = int(right - left)
            stages = block["split"][left:right]
            if stages.min() != stages.max():
                cross_groups += 1
                cross_rows += length
            else:
                item = per_split[NAMES[int(stages[0])]]
                item["groups"] += 1
                item["extra_rows"] += length - 1
        start = stop
    values._mmap.close()
    return per_split, cross_groups, cross_rows


def preflight(parts, boundaries, total_rows, types):
    stats = {"rows": Counter(), "labels": Counter(), "min_ts": {}, "max_ts": {},
             "quarantine_overlap": 0, "record_order_errors": 0, "record_id_errors": 0}
    near = defaultdict(set)
    previous_rows = {}
    forbidden = quarantine_ids()
    with tempfile.TemporaryDirectory(prefix="split_audit_", dir=REPORT) as temp:
        key_path = Path(temp) / "keys.bin"
        with key_path.open("wb") as key_file:
            for number, path in enumerate(parts, start=1):
                record_key_batch(path, {"keys": key_file}, boundaries, stats, near, forbidden, previous_rows)
                if number % 25 == 0:
                    print(f"Preflight: {number}/{len(parts)} parts", flush=True)
        if key_path.stat().st_size != total_rows * KEY_DTYPE.itemsize:
            raise ValueError("Duplicate audit did not receive every clean row")
        duplicates, cross_groups, cross_rows = audit_duplicate_keys(key_path, total_rows)
    stats["duplicates"] = duplicates
    stats["duplicate_cross_groups"] = cross_groups
    stats["duplicate_cross_rows"] = cross_rows
    stats["near_connections"] = [len(near[(i, "before")] & near[(i, "after")]) for i in range(2)]
    checks = {
        "row_conservation": sum(stats["rows"].values()) == total_rows,
        "time_order": all(name in stats["min_ts"] for name in NAMES) and stats["max_ts"]["train"] < stats["min_ts"]["validation"] and stats["max_ts"]["validation"] < stats["min_ts"]["test"],
        "both_labels": all(sum(count for (stage, label, _), count in stats["labels"].items() if stage == name and label == value) > 0 for name in NAMES for value in ("0", "1")),
        "attack_coverage": all({kind for (stage, _, kind), count in stats["labels"].items() if stage == name and kind != "normal" and count} == PROTOCOL[i] for i, name in enumerate(NAMES)),
        "duplicate_overlap": cross_groups == 0,
        "quarantine_exclusion": stats["quarantine_overlap"] == 0,
        "record_ids": stats["record_order_errors"] == stats["record_id_errors"] == 0,
        "label_conservation": all(sum(count for (_, _, observed_type), count in stats["labels"].items() if observed_type == kind) == profile["count"] for kind, profile in types.items()),
    }
    return stats, checks


def export_parts(parts, boundaries, source_schema):
    """Copy source rows unchanged into a staging tree; split UID by matching row."""
    if FINAL.exists() or STAGING.exists():
        raise FileExistsError("Split output or staging directory already exists")
    STAGING.mkdir(parents=True)
    for number, part in enumerate(parts, start=1):
        table = pq.read_table(part)
        ts = table.column("ts").to_numpy()
        groups = np.searchsorted(boundaries, ts, side="left")
        relative = part.relative_to(NETWORK)
        uid_path = UID / relative
        uid = pq.read_table(uid_path) if uid_path.exists() else None
        if uid is not None and (uid.num_rows != table.num_rows or not uid.column("record_id").equals(table.column("record_id"))):
            raise ValueError(f"UID sidecar does not align with {part}")
        for index, name in enumerate(NAMES):
            mask = groups == index
            if not mask.any():
                continue
            write_parquet(table.filter(pa.array(mask)).to_pandas(), STAGING / name / relative)
            if uid is not None:
                write_parquet(uid.filter(pa.array(mask)).to_pandas(), STAGING / "metadata" / "uid" / name / relative)
        if number % 25 == 0:
            print(f"Exported: {number}/{len(parts)} parts", flush=True)
    return source_schema


def validate_export(schema, stats, root=STAGING, boundaries=None):
    """Read back each output part and compare source columns, rows and labels."""
    rows = Counter()
    labels = Counter()
    ranges = {}
    schema_ok = trace_ok = True
    for name in NAMES:
        for part in (root / name).rglob("*.parquet"):
            parquet = pq.ParquetFile(part)
            rows[name] += parquet.metadata.num_rows
            schema_ok &= parquet.schema_arrow.names == schema.names and all(
                parquet.schema_arrow.field(name).type == schema.field(name).type for name in schema.names
            )
            subset = pq.read_table(part, columns=["ts", "label", "type", "record_id", "source_row", "source_file"])
            frame = subset.to_pandas()
            if frame["record_id"].isna().any():
                trace_ok = False
            for (label, kind), count in frame.groupby(["label", "type"]).size().items():
                labels[(name, str(label), str(kind))] += int(count)
            low, high = int(frame["ts"].min()), int(frame["ts"].max())
            ranges[name] = (min(ranges.get(name, (low, high))[0], low), max(ranges.get(name, (low, high))[1], high))
    uid_input = sum(pq.ParquetFile(path).metadata.num_rows for path in UID.rglob("*.parquet"))
    uid_output = sum(pq.ParquetFile(path).metadata.num_rows for path in (root / "metadata" / "uid").rglob("*.parquet"))
    checks = {
        "output_rows": rows == stats["rows"],
        "output_labels": labels == stats["labels"],
        "output_schema": schema_ok,
        "output_trace": trace_ok,
        "output_time_order": all(name in ranges for name in NAMES) and ranges["train"][1] < ranges["validation"][0] and ranges["validation"][1] < ranges["test"][0],
        "uid_preserved": uid_input == uid_output,
    }
    if boundaries:
        checks["output_boundaries"] = all(name in ranges for name in NAMES) and (
            ranges["train"][1] <= boundaries[0] < ranges["validation"][0]
            and ranges["validation"][1] <= boundaries[1] < ranges["test"][0]
        )
    return checks


def reports(types, daily, label_daily, boundaries, stats, checks, reasons, total_rows):
    REPORT.mkdir(parents=True, exist_ok=True)
    time_rows = [{"scope": "attack_type", "name": kind, "count": item["count"], "first_ts": item["first"], "last_ts": item["last"],
                  "first_utc": utc(item["first"]), "last_utc": utc(item["last"])} for kind, item in sorted(types.items())]
    time_rows += [{"scope": "day_type", "name": f"{utc(day * 86400)[:10]}:{kind}", "count": count, "first_ts": "", "last_ts": "", "first_utc": "", "last_utc": ""}
                  for (day, kind), count in sorted(daily.items())]
    time_rows += [{"scope": "day_label", "name": f"{utc(day * 86400)[:10]}:{label}", "count": count, "first_ts": "", "last_ts": "", "first_utc": "", "last_utc": ""}
                  for (day, label), count in sorted(label_daily.items())]
    write_csv(REPORT / "time_profile.csv", time_rows, ["scope", "name", "count", "first_ts", "last_ts", "first_utc", "last_utc"])
    distribution = []
    coverage = []
    for name in NAMES:
        total = stats["rows"].get(name, 0) if stats else 0
        for (stage, label, kind), count in sorted(stats["labels"].items() if stats else []):
            if stage != name:
                continue
            distribution.append({"split": name, "label": label, "type": kind, "count": count,
                                 "share_pct": round(100 * count / total, 6) if total else 0})
            if kind != "normal":
                coverage.append({"split": name, "type": kind, "count": count, "seen_in_train": kind in TRAIN_TYPES})
    write_csv(REPORT / "split_distribution.csv", distribution, ["split", "label", "type", "count", "share_pct"])
    write_csv(REPORT / "attack_type_coverage.csv", coverage, ["split", "type", "count", "seen_in_train"])
    write_csv(REPORT / "leakage_check.csv", [{"check": key, "result": "PASS" if value else "FAIL"} for key, value in checks.items()], ["check", "result"])
    lines = ["# Chia dữ liệu ToN-IoT Network", "", f"Nguồn sạch: {total_rows:,} dòng; cách ly không tham gia; excluded_rows = 0.",
             "Giao thức đánh giá: phát hiện nhị phân, train các tấn công sớm; validation chỉ Password; test là XSS, Ransomware, Backdoor, MITM."]
    if boundaries:
        lines += [f"Ranh giới train ≤ {boundaries[0]} ({utc(boundaries[0])}); validation ≤ {boundaries[1]} ({utc(boundaries[1])}); test lớn hơn ranh giới validation.",
                  f"Khoảng cách giữa hai giai đoạn tấn công: {min(types[k]['first'] for k in VALIDATION_TYPES) - boundaries[0]:,} giây và {min(types[k]['first'] for k in TEST_TYPES) - boundaries[1]:,} giây."]
    lines += attack_time_context(types)
    if stats:
        for name in NAMES:
            total = stats["rows"].get(name, 0)
            normal = sum(count for (stage, label, _), count in stats["labels"].items() if stage == name and label == "0")
            attack = total - normal
            kinds = sorted(kind for (stage, _, kind), count in stats["labels"].items() if stage == name and kind != "normal" and count)
            span = f"{utc(stats['min_ts'][name])} → {utc(stats['max_ts'][name])}" if name in stats["min_ts"] else "không có dữ liệu"
            lines += ["", f"## {name.title()}", "", f"{total:,} dòng; normal {normal:,} ({100 * normal / total:.3f}%); attack {attack:,}; kiểu: {', '.join(kinds)}; thời gian {span}.",
                      f"Nhóm trùng {stats['duplicates'][name]['groups']:,}; dòng trùng bổ sung {stats['duplicates'][name]['extra_rows']:,}."]
        lines += ["", f"Nhóm trùng xuyên tập: {stats['duplicate_cross_groups']:,}; dòng liên quan: {stats['duplicate_cross_rows']:,}.",
                  f"Kết nối giống nhau trong phạm vi 1 giờ hai phía ranh giới: {stats['near_connections'][0]:,} và {stats['near_connections'][1]:,}. Đây không đồng nghĩa rò rỉ bản ghi nguyên vẹn."]
    if reasons:
        lines += ["", "## Điều kiện không đạt", "", *[f"- {reason}" for reason in reasons]]
    lines += ["", "## Phạm vi kiểm tra", "", "Đã đối chiếu thời gian thật, nhãn, mã bản ghi, nhóm trùng hai băm, khu cách ly và UID. Không tính đặc trưng, chuẩn hóa hoặc cân bằng dữ liệu.",
              "Chưa có mã đợt tấn công đáng tin cậy; các kết nối lặp gần ranh giới không chứng minh các đợt hoàn toàn độc lập. Mã truy vết và khóa trùng không được dùng làm đặc trưng mô hình.",
              "Ba mô hình ML và GNN cần dùng đúng các tập này; việc tạo đồ thị/đặc trưng ở bước sau phải tránh đưa thông tin tương lai vào tập quá khứ.",
              "", "## Kiểm tra", "", *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()],
              "", f"Kết quả cuối: **{'PASS' if not reasons and all(checks.values()) else 'FAIL'}**."]
    (REPORT / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("========== NETWORK DATA SPLITTING ==========")
    print(f"Input rows: {total_rows:,}")
    if boundaries:
        print(f"Train <= {boundaries[0]} ({utc(boundaries[0])}); validation <= {boundaries[1]} ({utc(boundaries[1])}); test > {boundaries[1]}")
    if stats:
        for name in NAMES:
            print(f"{name}: {stats['rows'].get(name, 0):,} rows, {stats['duplicates'][name]['extra_rows']:,} duplicate extra rows")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(f"FINAL STATUS: {'PASS' if not reasons and all(checks.values()) else 'FAIL'}")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    parts, schema, total_rows = source_parts()
    types, daily, label_daily = profile_time(parts)
    boundaries, reasons = choose_boundaries(types)
    if boundaries is None or reasons:
        reports(types, daily, label_daily, boundaries, None, {}, reasons, total_rows)
        return 1
    stats, checks = preflight(parts, boundaries, total_rows, types)
    if not all(checks.values()):
        reports(types, daily, label_daily, boundaries, stats, checks, ["Preflight failed; no official split exported"], total_rows)
        return 1
    export_parts(parts, boundaries, schema)
    checks.update(validate_export(schema, stats, boundaries=boundaries))
    if all(checks.values()):
        STAGING.rename(FINAL)
    else:
        reasons.append("Output validation failed; staging was not promoted to official split")
    reports(types, daily, label_daily, boundaries, stats, checks, reasons, total_rows)
    return 0 if not reasons and all(checks.values()) else 1


def validate_only():
    """Re-read the promoted split after an interrupted terminal report."""
    _, schema, source_rows = source_parts()
    if not FINAL.exists():
        raise FileNotFoundError("Official split directory is missing")
    with (REPORT / "split_distribution.csv").open(encoding="utf-8-sig") as handle:
        labels = Counter({(row["split"], row["label"], row["type"]): int(row["count"]) for row in csv.DictReader(handle)})
    rows = Counter()
    for (name, _, _), count in labels.items():
        rows[name] += count
    with (REPORT / "split_boundaries.csv").open(encoding="utf-8-sig") as handle:
        saved = {row["boundary"]: int(row["ts"]) for row in csv.DictReader(handle)}
    boundaries = saved["train_end"], saved["validation_end"]
    checks = validate_export(schema, {"rows": rows, "labels": labels}, root=FINAL, boundaries=boundaries)
    checks["row_conservation"] = sum(rows.values()) == source_rows
    with (REPORT / "leakage_check.csv").open(encoding="utf-8-sig") as handle:
        original_checks = {row["check"]: row["result"] == "PASS" for row in csv.DictReader(handle)}
    checks["preflight_report"] = all(original_checks.values())
    write_csv(REPORT / "leakage_check.csv", [{"check": key, "result": "PASS" if value else "FAIL"}
                                             for key, value in {**original_checks, **checks}.items()], ["check", "result"])
    summary_path = REPORT / "split_summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("Đọc lại đầu ra chính thức: PASS.\n\n", "")
    summary = summary.replace("Đọc lại đầu ra chính thức: FAIL.\n\n", "")
    if "Khoảng giữa các đợt tấn công theo thời điểm đầu/cuối:" not in summary:
        with (REPORT / "time_profile.csv").open(encoding="utf-8-sig") as handle:
            types = {row["name"]: {"first": int(row["first_ts"]), "last": int(row["last_ts"])}
                     for row in csv.DictReader(handle) if row["scope"] == "attack_type"}
        summary = summary.replace("\n## Phạm vi kiểm tra", "\n" + "\n".join(attack_time_context(types)) + "\n\n## Phạm vi kiểm tra")
    summary = summary.replace("Kết quả cuối: **PASS**.",
                              "Đọc lại đầu ra chính thức: " + ("PASS" if all(checks.values()) else "FAIL") + ".\n\nKết quả cuối: **" + ("PASS" if all(checks.values()) else "FAIL") + "**.")
    summary_path.write_text(summary, encoding="utf-8")
    print("========== NETWORK DATA SPLITTING ==========")
    print(f"Input rows: {source_rows:,}; train: {rows['train']:,}; validation: {rows['validation']:,}; test: {rows['test']:,}")
    print(f"Train <= {boundaries[0]}; validation <= {boundaries[1]}; test > {boundaries[1]}")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(f"FINAL STATUS: {'PASS' if all(checks.values()) else 'FAIL'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(validate_only() if "--validate-only" in sys.argv else main())
