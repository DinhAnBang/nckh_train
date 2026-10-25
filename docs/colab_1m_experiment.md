# Thí nghiệm Network 1 triệu dòng trên Google Colab

## Trạng thái triển khai

Đã mở rộng hệ thống hiện có trong `src/ton_iot`, không tạo hệ thống huấn
luyện song song. Chế độ mới là `network_1m`; mỗi lần chạy chỉ nhận một mô
hình (`rf`, `xgboost` hoặc `lightgbm`). Không chạy huấn luyện chính thức trên
Windows.

Các file chính:

- `src/ton_iot/training/experiments.py`: tạo và kiểm tra bộ mẫu.
- `src/ton_iot/training/data_loader.py`: layout mẫu, CSR theo phần và bảo toàn
  giá trị 0.
- `src/ton_iot/training/train.py`: cờ `--experiment network_1m`, giới hạn
  luồng, kiểm tra GPU, giám sát tài nguyên và baseline.
- `src/ton_iot/training/evaluate.py`: F1 cho từng lớp và baseline luôn dự đoán
  attack.
- `notebooks/colab_network_1m.ipynb`: hướng dẫn chạy trên Colab.

## Cách lấy mẫu

Nguồn là `dataPreprocessing/outputs/features`, không đọc lại 23 CSV và không
fit lại quy tắc đặc trưng. Ba tập chính thức được xử lý độc lập:

| Tập | Dữ liệu nguồn | Mẫu |
|---|---:|---:|
| Train | 17.599.039 | 800.000 |
| Validation | 1.755.447 | 100.000 |
| Test | 2.983.666 | 100.000 |

Chương trình dùng seed `20261025`, reservoir sampling phân tầng theo loại
tấn công và 10 khoảng thời gian trong từng tập. Thứ tự ba tập không bị trộn;
X, y và metadata được lấy cùng vị trí dòng. Các loại có trong dữ liệu nguồn
được yêu cầu xuất hiện trong mẫu; MITM trong test được kiểm tra riêng.

Lệnh tạo mẫu trên Windows:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m ton_iot.training.experiments sample --seed 20261025
python -m ton_iot.training.experiments validate `
  --data-dir dataPreprocessing\outputs\experiments\network_1m `
  --report dataPreprocessing\outputs\experiments\network_1m\validation_report.json
```

Bộ mẫu thực tế đã tạo và kiểm tra PASS:

- Train: 800.000 dòng; normal 21.178; attack 778.822.
- Validation: 100.000 dòng; normal 2.102; attack 97.898.
- Test: 100.000 dòng; normal 9.812; attack 90.188.
- Test có MITM: 35 dòng.
- Tổng dung lượng bộ mẫu hiện tại: 62.450.168 bytes, khoảng 0,058 GiB; chưa
  tính các model/metrics sinh ra sau huấn luyện.
- 197 đặc trưng; X/y/metadata khớp; nhãn chỉ 0/1; không có X thiếu hoặc vô
  hạn; thứ tự thời gian giữa ba tập đạt.
- Có 112 nhóm và 540 dòng cùng vector 197 đặc trưng xuyên tập. Đây là thông
  tin kiểm tra, không bị xóa.

Bộ mẫu và validation report nằm dưới `dataPreprocessing/outputs/`, đã được
`.gitignore` loại khỏi GitHub.

## Chạy từng mô hình

Sau khi bộ mẫu được đưa lên Google Drive và sao chép về `/content`, chạy một
mô hình mỗi lần:

```bash
python -m ton_iot.training.train \
  --experiment network_1m \
  --model rf \
  --train-mode full \
  --data-dir /content/ton_iot_network/colab_data/network_1m \
  --output-dir /content/drive/MyDrive/ton_iot_network/results/network_1m/rf \
  --device cpu \
  --cpu-threads 2 \
  --max-ram-gib 10
```

Thay `rf` bằng `xgboost` hoặc `lightgbm`. Không dùng `--model all` cho
`network_1m`; chạy riêng giúp giảm rủi ro tích lũy RAM và cho phép lưu kết quả
sau từng mô hình. Output chứa model, `run_manifest.json`, metrics và recall
theo loại tấn công. Metrics cũng có dòng `baseline_always_attack`.

## GPU XGBoost

Không mặc định rằng CUDA hoạt động. Sau khi Colab có GPU T4, chạy bài thử nhỏ:

```bash
python -m ton_iot.training.train \
  --experiment network_1m \
  --model xgboost \
  --gpu-check \
  --device gpu \
  --data-dir /content/ton_iot_network/colab_data/network_1m \
  --cpu-threads 2
```

Nếu bài thử thất bại, dùng `--device cpu` và không xem GPU là đã được xác
nhận. Random Forest chạy CPU; LightGBM mặc định chạy CPU. XGBoost được chuyển
từ CSR sang dense trước khi fit để giá trị 0 vẫn là số 0, không bị hiểu thành
giá trị thiếu.

## Tài nguyên và giới hạn

Trên bộ mẫu, chương trình đo được ước lượng khoảng 1,17 GiB cho ma trận train
dense float64 và 0,22 GiB cho CSR. Đây chỉ là kích thước ma trận, chưa bao
gồm cây mô hình, thư viện và hệ điều hành. Bộ chặn RAM vẫn được dùng và có thể
giảm bằng `--max-ram-gib`. Mỗi lần fit có `peak_rss_gib`, GPU memory peak nếu
`nvidia-smi` hoạt động, thời gian fit và phiên bản thư viện trong manifest.

Các smoke test Windows đã chạy thành công cho cả ba model với 2.000 dòng
train và 400 dòng validation; đã kiểm tra lưu/tải model và metrics. GPU T4
chưa được kiểm thử trên Windows vì máy phát triển không có `nvidia-smi`; bài
kiểm tra GPU nằm trong notebook/CLI để chạy trực tiếp trên Colab.

## Quy trình notebook

Mở `notebooks/colab_network_1m.ipynb` và thực hiện lần lượt:

1. Kiểm tra CPU, RAM, GPU, ổ đĩa và phiên bản thư viện.
2. Kết nối Google Drive.
3. Điền URL repository và clone branch cần dùng.
4. Cài `requirements.txt` và package editable.
5. Chọn thư mục sample trên Drive, copy vào `/content`.
6. Validate sample.
7. Tùy chọn kiểm tra GPU XGBoost.
8. Chọn đúng một model và chạy; output lưu thẳng về Drive.

Notebook không chứa thuật toán huấn luyện, không ghi cứng URL GitHub và không
tự động chạy cả ba model.

## GitHub

Dataset, Parquet sample, model, backup, reports sinh ra và cache đều nằm trong
`.gitignore`. Chỉ commit notebook, mã nguồn `src`, tài liệu, requirements và
`pyproject.toml`. Không đặt token hoặc thông tin Drive vào notebook trước khi
đưa lên GitHub.
