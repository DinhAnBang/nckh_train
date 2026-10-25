# Báo cáo sắp xếp lại mã nguồn

## Phạm vi

Đợt này sắp xếp lại mã nguồn để dễ đọc và cập nhật, sau đó chạy lại đầy đủ
pipeline bằng package mới để tạo output mới. Không đổi quy tắc làm sạch/chia
tập/chọn đặc trưng và không huấn luyện mô hình. Dataset gốc và backup nằm
ngoài phạm vi di chuyển.

Nhánh thực hiện: `refactor/unified-src`.

## Cấu trúc mới

```text
src/ton_iot/
├── config/
│   ├── paths.py       # đường dẫn dự án dùng chung
│   └── training.py    # cấu hình ba mô hình
├── data/
│   ├── reader.py      # bước 1
│   ├── analyzer.py    # bước 2: phân tích
│   ├── verifier.py    # bước 2: xác minh
│   ├── cleaner.py     # bước 3
│   ├── splitter.py    # bước 4
│   └── preprocessor.py # bước 5
├── training/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── train.py       # bước 6
│   ├── utils.py
│   └── models/
├── validation/
│   └── audit.py       # kiểm tra độc lập output
├── reporting/
│   └── reports.py
└── pipeline/
    ├── backup.py
    └── runner.py
```

Các file mã nguồn cũ trong `dataPreprocessing`, `modelTraining` và
`pipeline_tools` đã được xóa theo yêu cầu sau khi kiểm tra bản chính trong
`src/ton_iot`. Mã cần cập nhật hiện chỉ nằm trong `src/ton_iot`.

## Ánh xạ vị trí cũ sang vị trí mới

| Vị trí cũ | Vị trí chính mới |
|---|---|
| `dataPreprocessing/reader.py` | `src/ton_iot/data/reader.py` |
| `dataPreprocessing/analyzer.py` | `src/ton_iot/data/analyzer.py` |
| `dataPreprocessing/verify_issues.py` | `src/ton_iot/data/verifier.py` |
| `dataPreprocessing/cleaner.py` | `src/ton_iot/data/cleaner.py` |
| `dataPreprocessing/splitter.py` | `src/ton_iot/data/splitter.py` |
| `dataPreprocessing/preprocessor.py` | `src/ton_iot/data/preprocessor.py` |
| `modelTraining/config.py` | `src/ton_iot/config/training.py` |
| `modelTraining/data_loader.py` | `src/ton_iot/training/data_loader.py` |
| `modelTraining/evaluate.py` | `src/ton_iot/training/evaluate.py` |
| `modelTraining/train.py` | `src/ton_iot/training/train.py` |
| `modelTraining/utils.py` | `src/ton_iot/training/utils.py` |
| `modelTraining/models/*` | `src/ton_iot/training/models/*` |
| `pipeline_tools/backup_outputs.py` | `src/ton_iot/pipeline/backup.py` |
| `pipeline_tools/run_steps.py` | `src/ton_iot/pipeline/runner.py` |
| `pipeline_tools/audit_dataset.py` | `src/ton_iot/validation/audit.py` |
| `pipeline_tools/make_reports.py` | `src/ton_iot/reporting/reports.py` |
| `modelTraining/README.md` | `docs/training.md` |

## Những điểm đã cải thiện

- Đường dẫn dự án được tính một lần trong `config/paths.py`, không phụ thuộc
  vào thư mục đang mở terminal.
- Các import giữa các bước dùng package rõ ràng, không còn phụ thuộc vào
  việc chạy file từ đúng thư mục chứa file đó.
- Tách rõ mã dữ liệu, mã huấn luyện, kiểm tra độc lập, báo cáo và điều phối.
- Thêm `pyproject.toml` để có thể cài package editable bằng `python -m pip
  install -e .`; `requirements.txt` vẫn là nơi cài các thư viện nghiên cứu.
- Xóa các launcher/mã nguồn cũ ngoài `src` sau khi xác minh bản chính đã đầy
  đủ trong package mới.

## Kiểm tra đã thực hiện

| Kiểm tra | Kết quả |
|---|---|
| `compileall` cho package mới | Đạt |
| Import toàn bộ module chính trong `src/ton_iot` | Đạt |
| Kiểm tra không còn file `.py` cũ ngoài `src` | Đạt |
| Tìm nguồn Network | Đạt: đúng `Processed_datasets/Processed_Network_dataset`, 23 file |
| Backup cũ | Đạt; manifest xác nhận `verified=True` |
| `python -m ton_iot.training.train --model all --dry-run` | Đạt; không fit mô hình |
| Kiểm tra output hiện có qua `audit_stats.json` | Đạt; `issues={}` |

Dry-run sau refactor đọc output hiện có và xác nhận:

- Train: 17.599.039 dòng.
- Validation: 1.755.447 dòng.
- Test: 2.983.666 dòng.
- 197 đặc trưng.
- Nhãn và X/y/metadata khớp theo các phần Parquet.
- Không có mô hình được fit.

## Dữ liệu và kết quả được giữ nguyên

Không di chuyển hoặc xóa:

- `Raw_datasets/` và `Processed_datasets/`.
- `dataPreprocessing/outputs/` và `dataPreprocessing/reports/` hiện tại.
- `reports/pipeline_20261025_112206/`.
- Backup ngoài repository tại `D:\paper_NCKH\pipeline_backups\pipeline_20261025_112206`.
- Output huấn luyện trong `modelTraining/outputs/`.

Backup đã được xác minh trước đó bằng SHA-256 cho 1.816 file. `.gitignore`
tiếp tục loại trừ dataset lớn, output, report và model output; mã nguồn,
`requirements.txt`, `pyproject.toml` và tài liệu vẫn được phép đưa lên GitHub.

## Các kết quả dữ liệu cần tiếp tục ghi nhớ

Đây là kết quả của lần chạy pipeline mới sau refactor:

- Tổng dữ liệu sạch: 22.338.152 dòng; 869 dòng bị cách ly.
- Output features: 197 cột.
- Có 13.743 nhóm và 66.354 dòng có cùng vector 197 đặc trưng xuyên tập; không
  tự xóa dữ liệu này.
- Output audit có trạng thái `Đạt`, không có lỗi cấu trúc/giá trị trong
  `issues`, nhưng cảnh báo trùng vector đặc trưng xuyên tập vẫn phải được
  nêu trong báo cáo nghiên cứu.
- Ước lượng train dense float64 khoảng 25,83 GiB; CSR khoảng 4,37 GiB. Máy
  hiện tại có khoảng 15,69 GiB RAM tổng và khoảng 2,6 GiB khả dụng, vì vậy
  chưa chạy full training.
- Output/cache cũ được chuyển nguyên vẹn sang
  `D:\paper_NCKH\pipeline_backups\cleanup_20261025_134052` trước khi chạy.

## Chưa làm trong đợt này

- Chưa huấn luyện Random Forest, XGBoost hoặc LightGBM; bước 6 chỉ dry-run.
- Chưa thay đổi quy tắc xử lý dữ liệu, ranh giới chia tập, thứ tự nhãn hay
  lựa chọn đặc trưng.
- Chưa cài package editable và chưa push lên GitHub.

Lệnh chạy ưu tiên từ thư mục gốc sau khi cài môi trường:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ton_iot.training.train --model all --dry-run
```
