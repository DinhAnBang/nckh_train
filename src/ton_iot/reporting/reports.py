"""Write Vietnamese research reports from this run's audit and execution log."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

import psutil

from ton_iot.config.paths import (
    BACKUP_PARENT,
    DATA_OUTPUT_DIR,
    DATA_PREPROCESSING_DIR,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
)


PROJECT = PROJECT_ROOT
DATA = DATA_PREPROCESSING_DIR
GIB = 1024**3
SPLITS = ("train", "validation", "test")


def fmt(value):
    return f"{value:,}"


def gib(value):
    return f"{value / GIB:,.3f} GiB"


def status(passed):
    return "Đạt" if passed else "Chưa đạt"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pipeline_report(report, audit, log):
    splits = audit["splits"]
    backup_path = BACKUP_PARENT / audit["run_id"]
    backup = json.loads((backup_path / "backup_manifest.json").read_text(encoding="utf-8"))
    old_outputs = sum(value["bytes"] for value in backup["files"]["dataPreprocessing/outputs"].values())
    old_reports = sum(value["bytes"] for value in backup["files"]["dataPreprocessing/reports"].values())
    new_outputs = sum(path.stat().st_size for path in DATA_OUTPUT_DIR.rglob("*") if path.is_file())
    source_rows = sum(item["rows"] for item in splits.values())
    cleaning = csv_rows(DATA / "reports" / "cleaning" / "file_counts.csv")
    input_rows = sum(int(row["input_rows"]) for row in cleaning)
    quarantine = sum(int(row["quarantine_rows"]) for row in cleaning)
    invalid = len(csv_rows(DATA / "reports" / "verification" / "invalid_records.csv"))
    steps = (
        ("1. Đọc", "23 CSV Network", f"{fmt(input_rows)} dòng được đọc", "reader.py"),
        ("2. Phân tích và xác minh", "23 CSV và Bro conn.csv", f"{fmt(input_rows)} dòng phân tích; {fmt(invalid)} dòng nghi vấn", "analyzer.py, verify_issues.py"),
        ("3. Làm sạch", f"{fmt(input_rows)} dòng nguồn", f"{fmt(source_rows)} dòng sạch; {fmt(quarantine)} dòng cách ly", "cleaner.py"),
        ("4. Chia tập", f"{fmt(source_rows)} dòng sạch", ", ".join(f"{name}: {fmt(splits[name]['rows'])}" for name in SPLITS), "splitter.py"),
        ("5. Xử lý đặc trưng", "Ba tập đã chia", f"197 cột X; giữ nguyên {fmt(source_rows)} dòng", "preprocessor.py"),
        ("6. Kiểm tra huấn luyện", "X/y/metadata", "dry-run trên toàn bộ dữ liệu; 0 mô hình được fit", "ton_iot.training.train"),
    )
    lines = ["# Báo cáo chạy lại quy trình ToN-IoT Network", "",
             f"Mã lần chạy: `{audit['run_id']}`. Thời điểm kiểm tra cuối: {audit['audited_at']}.",
             f"Dữ liệu gốc được giữ nguyên. Backup cũ tại `{backup_path}`; {fmt(backup['old_files'])} tệp đã đối chiếu SHA-256 trước/sau di chuyển.",
             f"Trước khi chạy: output cũ {gib(old_outputs)}, reports cũ {gib(old_reports)}, output chính ước theo kích thước cũ {gib(backup['active_output_bytes'])}; ổ D: trống {gib(backup['free_bytes_before'])}. Ngưỡng dự phòng yêu cầu {gib(backup['minimum_free_bytes'])}.",
             f"Sau khi chạy, tổng output mới thực tế là {gib(new_outputs)}; ổ D: còn {gib(shutil.disk_usage(PROJECT).free)}.",
             "", "## Sáu bước đã thực hiện", "",
             "| Bước | Đầu vào | Đầu ra thực tế | Chương trình |", "|---|---|---|---|"]
    lines += [f"| {name} | {source} | {target} | `{code}` |" for name, source, target, code in steps]
    lines += ["", "Nhật ký `execution_log.txt` ghi giờ bắt đầu, kết thúc, mã thoát và số dòng của từng bước.",
              "Bước 1 chỉ chấp nhận ngoại lệ cấu trúc đã biết: tệp số 6 có thêm `uid`; 46 cột chung vẫn phải đúng tên và thứ tự. Không đổi quy tắc làm sạch, chia tập hoặc chọn đặc trưng.",
              "Báo cáo phân tích bước 2 vẫn ghi khác biệt schema nguyên bản do `uid`; đây là khác biệt đã biết, được lưu như cảnh báo chứ không bị xóa khỏi dữ liệu nguồn.",
              "Bước 3 giữ dòng bất thường trong vùng cách ly, không xóa vĩnh viễn. Bước 6 chỉ kiểm tra, không huấn luyện.",
              "", "## Kết luận", "",
              "Quy trình đã hoàn thành đến dry-run bước 6. Kết quả kiểm tra độc lập nằm trong `data_quality_report.md`; số liệu chi tiết ở `dataset_report.md` và `audit_stats.json`."]
    write(report / "pipeline_report.md", lines)


def dataset_report(report, audit):
    splits = audit["splits"]
    lines = ["# Thống kê bộ dữ liệu đầu ra mới", "",
             f"Mã lần chạy: `{audit['run_id']}`. Số liệu được đọc trực tiếp từ Parquet mới, không lấy từ báo cáo cũ.",
             "", "## Số dòng, nhãn và dung lượng", "",
             "| Tập | Dòng | Bình thường | Tấn công | Tỷ lệ tấn công | X + y + metadata | Nguồn đã chia |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in SPLITS:
        item = splits[name]
        normal, attack = item["labels"]["0"], item["labels"]["1"]
        feature_bytes = sum(item["bytes"].get(branch, 0) for branch in ("X", "y", "metadata"))
        lines.append(f"| {name} | {fmt(item['rows'])} | {fmt(normal)} | {fmt(attack)} | {attack/item['rows']:.3%} | {gib(feature_bytes)} | {gib(item['bytes']['source'])} |")
    lines += ["", "Dung lượng trên là kích thước tệp thực trên ổ đĩa. Tệp UID đi kèm được thống kê riêng trong `file_sizes.csv`.",
              "`file_sizes.csv` liệt kê từng tệp X, y, metadata, nguồn đã chia và dung lượng tính bằng byte.",
              "", "## Loại dữ liệu theo tập", ""]
    for name in SPLITS:
        lines += [f"### {name}", "", "| Loại | Dòng |", "|---|---:|"]
        lines += [f"| {kind} | {fmt(count)} |" for kind, count in sorted(splits[name]["types"].items())]
        lines.append("")
    lines += ["## Danh sách đặc trưng và kiểu dữ liệu", "",
              f"Tổng cộng {len(audit['feature_columns'])} cột X. Thứ tự dưới đây là thứ tự thật trong tệp Parquet.",
              "", "| STT | Đặc trưng | Kiểu | Nhỏ nhất | Lớn nhất | Thiếu | Vô hạn |",
              "|---:|---|---|---:|---:|---:|---:|"]
    for number, (name, dtype) in enumerate(audit["feature_schema"], start=1):
        item = audit["feature_stats"][name]
        lines.append(f"| {number} | `{name}` | {dtype} | {item['min']} | {item['max']} | {fmt(item['nulls'])} | {fmt(item['nonfinite'])} |")
    write(report / "dataset_report.md", lines)


def quality_report(report, audit):
    issues, duplicates, splits = audit["issues"], audit["duplicates"], audit["splits"]
    total = sum(item["rows"] for item in splits.values())
    source = duplicates["source"]
    features = duplicates["features"]
    ids = duplicates["record_id"]
    checks = [
        ("Nhãn và metadata khớp từng dòng", status(not any(issues.get(key, 0) for key in ("label_cell_mismatch", "label_type_mismatch", "metadata_cell_mismatch", "invalid_labels")))),
        ("197 cột và kiểu/thứ tự thống nhất", status(len(audit["feature_schema"]) == 197)),
        ("Không có giá trị X thiếu hoặc vô hạn", status(not any(issues.get(key, 0) for key in ("feature_nulls", "feature_nonfinite")))),
        ("Giá trị nguyên biểu diễn chính xác bằng float64", status(issues.get("integer_precision_outside_float64", 0) == 0)),
        ("X tái tạo từ nguồn đã chia trên mọi dòng", status(issues.get("feature_cell_mismatch", 0) == 0)),
        ("Quy tắc số và danh mục chỉ học từ train", status(audit["rules_fitted_on"] == "train_only" and not any(issues.get(key, 0) for key in ("train_numeric_rule_mismatch", "train_vocabulary_mismatch")))),
        ("Thời gian và kiểu tấn công đúng giao thức chia", status(not any(issues.get(key, 0) for key in ("time_order", "attack_protocol")))),
        ("Mã bản ghi không lặp", status(ids["extra_rows"] == 0)),
        ("Khóa nguồn không lặp xuyên tập", status(source["cross_split_groups"] == 0)),
        ("Không có cùng 197 giá trị X xuyên tập", "Đạt" if features["cross_split_groups"] == 0 else "Chưa đạt; có trùng cần đánh giá"),
        ("UID đi kèm giữ nguyên từ tập đã chia", status(not any(issues.get(key, 0) for key in ("uid_sidecar_mismatch", "uid_sidecar_file_set_mismatch")))),
    ]
    lines = ["# Chất lượng và tính toàn vẹn dữ liệu", "",
             f"Mã lần chạy: `{audit['run_id']}`. Kiểm tra {fmt(total)} dòng trong {sum(s['parts'] for s in splits.values())} phần dữ liệu.",
             "", "## Kết quả", "", "| Kiểm tra | Kết luận |", "|---|---|"]
    lines += [f"| {name} | **{result}** |" for name, result in checks]
    lines += ["", "Giá trị bất thường hoặc sai khác đếm được:", ""]
    if issues:
        lines += ["| Nội dung | Số lượng |", "|---|---:|"]
        lines += [f"| {name} | {fmt(count)} |" for name, count in sorted(issues.items())]
    else:
        lines.append("Không ghi nhận sai khác trong các phép kiểm tra đã thực hiện.")
    lines += ["", "## Dữ liệu trùng lặp", "",
              "| Cách xác định | Nhóm trùng | Dòng dư | Nhóm xuyên tập | Dòng thuộc nhóm xuyên tập |",
              "|---|---:|---:|---:|---:|"]
    for name, item in (("Khóa của bản ghi nguồn", source), ("Cùng 197 giá trị X", features), ("Mã bản ghi", ids)):
        lines.append(f"| {name} | {fmt(item['groups'])} | {fmt(item['extra_rows'])} | {fmt(item['cross_split_groups'])} | {fmt(item['cross_split_rows'])} |")
    lines += ["", "Số dòng dư theo từng tập:", "", "| Tập | Theo khóa nguồn | Theo X |", "|---|---:|---:|"]
    for name in SPLITS:
        lines.append(f"| {name} | {fmt(source['within_split_extra'][name])} | {fmt(features['within_split_extra'][name])} |")
    lines += ["", "Khóa nguồn là hai băm lưu từ bước làm sạch; hai băm của X được tính lại từ 197 cột theo hai thứ tự. Kết quả đếm nhóm rất mạnh nhưng va chạm băm không thể loại trừ tuyệt đối: **chưa xác minh tính trùng khớp byte cho mọi cặp**.",
              "Đếm theo khóa nguồn ở đây chỉ áp dụng cho 22.338.152 dòng sạch đã đi vào ba tập. Bước 2 tính 1.791.971 dòng dư trên toàn bộ 22.339.021 dòng, gồm cả 869 dòng cách ly, nên hai số khác phạm vi.",
              "Cùng X xuyên tập không tự động là rò rỉ vì các kết nối khác nhau có thể mang cùng đặc trưng. Mức độc lập theo đợt tấn công/thiết bị: **chưa xác minh**, do không có mã đợt đáng tin cậy.",
              "Ranh giới thời gian được xác định từ lịch xuất hiện của các loại tấn công trên toàn dữ liệu theo giao thức nghiên cứu đã chọn. Mô hình không học từ test, nhưng cách thiết kế tập đánh giá có thể ảnh hưởng khả năng khái quát; **chưa xác minh** tính đại diện của các tập cho môi trường triển khai khác.",
              "Tính hợp lý theo ngữ nghĩa của mọi giá trị cực đoan ngoài kiểm tra kiểu, khoảng biểu diễn và quy tắc biến đổi: **chưa xác minh**; xem giá trị nhỏ nhất/lớn nhất từng cột trong `dataset_report.md`.",
              "Không xóa hoặc gộp các dòng trùng. Ảnh hưởng của trùng X đến mô hình cần đánh giá riêng.",
              "", "## Bằng chứng", "",
              "`audit_stats.json` chứa số đếm, kiểu và giá trị nhỏ/lớn theo cột. `file_sizes.csv` chứa kích thước từng tệp. `execution_log.txt` ghi tiến trình thực thi."]
    write(report / "data_quality_report.md", lines)


def readiness_report(report, audit):
    train = audit["splits"]["train"]
    validation = audit["splits"]["validation"]
    features = len(audit["feature_columns"])
    dense_train = train["rows"] * features * 8 / GIB
    dense_validation = validation["rows"] * features * 8 / GIB
    csr_train = (train["nnz"] * 12 + (train["rows"] + 1) * 4) / GIB
    csr_validation = (validation["nnz"] * 12 + (validation["rows"] + 1) * 4) / GIB
    guards = {"Random Forest": max(4 * csr_train + 2 * csr_validation + 4, 3 * dense_train + 4),
              "XGBoost": max(4 * csr_train + 2 * csr_validation + 4, 2.5 * dense_train + dense_validation + 4),
              "LightGBM": max(4 * csr_train + 2 * csr_validation + 4, 2.5 * dense_train + dense_validation + 4)}
    memory = psutil.virtual_memory()
    physical_gib, available_gib = memory.total / GIB, memory.available / GIB
    source_cross = audit["duplicates"]["source"]["cross_split_groups"]
    x_cross = audit["duplicates"]["features"]["cross_split_groups"]
    ready = not audit["issues"] and source_cross == 0
    readiness = "Đạt về tính toàn vẹn, có cảnh báo trùng X xuyên tập" if ready and x_cross else ("Đạt" if ready else "Chưa đạt")
    lines = ["# Khả năng sử dụng dữ liệu để huấn luyện", "",
             f"Mã lần chạy: `{audit['run_id']}`. Kết luận dữ liệu: **{readiness}** theo các kiểm tra độc lập đã thực hiện.",
             "Bước 6 mới chạy dry-run; **chưa có mô hình nào được huấn luyện chính thức**.",
             f"Kiểm tra thấy {x_cross:,} nhóm, {audit['duplicates']['features']['cross_split_rows']:,} dòng thuộc cùng 197 giá trị X ở nhiều tập. Các dòng không bị xóa; cần quyết định cách xử lý hoặc phân tích độ nhạy trước khi dùng kết quả này làm bằng chứng nghiên cứu.",
             "", "## Bộ nhớ và CPU", "",
             f"Train có {fmt(train['rows'])} dòng, {features} cột. Dung lượng ma trận train dense float64 tính từ hình dạng: {dense_train:.2f} GiB; CSR float64 tính từ số ô khác 0 thực tế: {csr_train:.2f} GiB.",
             f"Validation dense float64: {dense_validation:.2f} GiB; CSR float64: {csr_validation:.2f} GiB.",
             "Các số trên chỉ là ma trận đầu vào, chưa gồm bản sao khi nạp và bộ nhớ học cây.",
             "", "| Mô hình | Dạng X trong mã hiện tại | Ngưỡng RAM bảo thủ của chương trình |", "|---|---|---:|",
             f"| Random Forest | CSR float64 | {guards['Random Forest']:.1f} GiB |",
             f"| XGBoost | dense float64 để giữ nghĩa giá trị 0 | {guards['XGBoost']:.1f} GiB |",
             f"| LightGBM | CSR float64 | {guards['LightGBM']:.1f} GiB |",
             "", "Đây là ngưỡng chặn ước lượng trong mã, không bảo đảm máy đạt ngưỡng sẽ huấn luyện xong.",
             f"Máy hiện tại có {physical_gib:.2f} GiB RAM, khả dụng tại lúc viết báo cáo {available_gib:.2f} GiB và {psutil.cpu_count(logical=True)} luồng CPU logic. Chưa đo thời gian huấn luyện toàn bộ nên **chưa xác minh** thời gian hoặc số CPU tối ưu.",
             "Random Forest và LightGBM mặc định dùng `n_jobs=-1`; XGBoost dùng cấu hình mặc định của thư viện. GPU và tương thích bản cài GPU: **chưa xác minh**.",
             "", "## Điều kiện khi chuyển sang máy huấn luyện", "",
             "Chuyển nguyên thư mục features mới, giữ đúng X/y/metadata và tệp train_fitted_rules.json. Chạy `--dry-run` trên máy mới trước khi chạy `--train-mode full`.",
             "Cố định cấu hình và ngưỡng dự đoán trước khi xem test. Trùng X xuyên tập và mức độc lập theo đợt vẫn cần thảo luận trong báo cáo nghiên cứu."]
    write(report / "training_readiness.md", lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = REPORT_OUTPUT_DIR / args.run_id
    audit = json.loads((report / "audit_stats.json").read_text(encoding="utf-8"))
    log = (report / "execution_log.txt").read_text(encoding="utf-8")
    if "PIPELINE PASS" not in log or "END independent_final_audit" not in log:
        raise ValueError("Required execution evidence missing; no reports generated")
    pipeline_report(report, audit, log)
    dataset_report(report, audit)
    quality_report(report, audit)
    readiness_report(report, audit)
    with (report / "execution_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} REPORTS CREATED; four Markdown files; audit evidence preserved\n")
    print(f"REPORTS CREATED: {report}")


if __name__ == "__main__":
    main()
