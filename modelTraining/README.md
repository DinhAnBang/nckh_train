# Huấn luyện ToN-IoT Network (bước 6)

Module này dùng 197 đặc trưng đã tạo ở bước 5 để huấn luyện bộ phân loại nhị phân: `0 = normal`, `1 = attack`. Ba mô hình là Random Forest, XGBoost và LightGBM. `type` chỉ được dùng khi thống kê khả năng phát hiện từng kiểu tấn công; nó không nằm trong X.

## Chuẩn bị dữ liệu và môi trường

Sao chép cả mã nguồn dự án và thư mục `dataPreprocessing/outputs/features` sang máy chạy. Giữ cấu trúc `train`, `validation`, `test`, `metadata/train_fitted_rules.json`. Mỗi phần Parquet phải có cùng đường dẫn tương đối trong `X`, `y` và `metadata`. Chương trình không tải dữ liệu từ Internet. Có thể đặt dữ liệu ở nơi khác bằng `--data-dir`.

Chạy từ thư mục gốc dự án. Trên Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modelTraining.train --model all --dry-run
```

Trên Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m modelTraining.train --model all --dry-run
```

Các lệnh chạy dùng Python 3.12 đã kiểm thử tại máy phát triển. Ghi lại phiên bản thực tế từ `run_manifest.json` khi chạy trên máy khác. Với XGBoost GPU cần bản cài có CUDA phù hợp; LightGBM GPU cần bản build hỗ trợ GPU. `nvidia-smi` phải chạy được trước khi dùng `--device gpu` trong chương trình này. Thành công ở dry-run CPU không xác nhận GPU hoạt động.

## Chế độ chạy

```bash
python -m modelTraining.train --model all --dry-run
python -m modelTraining.train --model all --smoke-test --train-limit 1000 --validation-limit 200
python -m modelTraining.train --model rf --train-mode full
python -m modelTraining.train --model xgboost --train-mode full
python -m modelTraining.train --model lightgbm --train-mode full
python -m modelTraining.train --model all --train-mode full
```

Thay `--model all` bằng `rf`, `xgboost` hoặc `lightgbm` để chạy riêng. `--dry-run` đọc và kiểm tra các phần dữ liệu, nhãn, metadata, kiểu và giá trị đặc trưng, đồng thời in ước lượng tài nguyên; không fit mô hình. `--smoke-test` dùng mặc định 2.000 dòng train và 400 dòng validation, lấy số lượng bằng nhau từ hai lớp theo thứ tự phần dữ liệu. Có thể đổi kích thước bằng `--train-limit` và `--validation-limit`. Kết quả này chỉ kiểm tra kỹ thuật; mẫu có phân bố khác dữ liệu thật. `--train-limit` cùng với `--train-mode full` cũng tạo một lần chạy **technical**, không phải huấn luyện chính thức. Chỉ lệnh có `--train-mode full` mà không có `--train-limit` mới dùng toàn bộ train. Không có chế độ huấn luyện mặc định khi thiếu cờ chế độ.

Để chỉ định nơi chứa dữ liệu hoặc nơi lưu kết quả:

```bash
python -m modelTraining.train --model all --dry-run --data-dir /path/to/features --output-dir /path/to/results
```

Trên Windows dùng đường dẫn tương ứng, ví dụ `--data-dir E:\data\features`. Các đường dẫn mặc định được tính từ vị trí file mã nguồn, không phụ thuộc thư mục terminal. Chạy bằng `python -m ...` từ thư mục gốc để Python tìm được package.

Để dùng GPU cho một mô hình phù hợp, thay `--device cpu` bằng `--device gpu` khi chạy `xgboost` hoặc `lightgbm`; Random Forest trong chương trình này dùng CPU. Chạy thử smoke test GPU riêng từng mô hình trước khi chạy full. Nếu bộ thư viện không hỗ trợ GPU, cần cài bản phù hợp trên máy thuê và kiểm tra lại.

## Cấu hình mô hình

Giá trị mặc định ở `config.py`. Truyền một JSON chứa các tham số cần ghi đè; các khóa hợp lệ là tên tham số đã có trong cấu hình mặc định:

```json
{
  "rf": {"n_estimators": 150, "n_jobs": 8},
  "xgboost": {"n_estimators": 600, "max_depth": 5},
  "lightgbm": {"n_estimators": 600, "num_leaves": 31}
}
```

```bash
python -m modelTraining.train --model all --smoke-test --config training_config.json
```

Ngưỡng phân loại cố định là 0,5. Không tự cân bằng trọng số lớp; chỉ dùng `class_weight` khi người chạy cấu hình. Random seed mặc định là 42. XGBoost và LightGBM dùng validation để dừng sớm khi `--early-stopping-rounds` lớn hơn 0; test không tham gia fit, dừng sớm hoặc chọn ngưỡng.

## Tài nguyên và cách đọc dữ liệu

Chạy dry-run trên dữ liệu hiện tại cho thấy 17.599.039 dòng train, 1.755.447 dòng validation, 2.983.666 dòng test, 197 đặc trưng. Tổng dung lượng nén của X Parquet khoảng 0,23 GiB, nhưng riêng ma trận train dense float64 ước khoảng 25,83 GiB, CSR khoảng 4,37 GiB dựa trên một phần mẫu. Đây là **ước lượng**, chưa tính đầy đủ bộ nhớ nội bộ của bộ học cây, dữ liệu tạm và hệ điều hành. Không dùng dung lượng Parquet làm ước lượng RAM.

Random Forest và LightGBM nhận CSR float64. XGBoost nhận dense float64 vì ô 0 không được phép biến thành giá trị thiếu do cơ chế sparse của XGBoost. Việc chuyển đổi này cần nhiều RAM hơn. Chương trình nạp từng phần Parquet để tạo ma trận, nhưng cả ba bộ phân loại ở cấu hình hiện tại đều fit trên ma trận train trong bộ nhớ. Không có học tăng dần hoặc out-of-core. Full mode có bộ chặn dựa trên RAM khả dụng và ước lượng bảo thủ; vượt qua bộ chặn không bảo đảm fit thành công. Kiểm tra RAM, ổ đĩa, GPU và dùng smoke test trước khi chạy full:

```bash
python -m modelTraining.train --model all --dry-run
nvidia-smi
```

Có thể đặt trần RAM thấp hơn RAM hệ thống bằng `--max-ram-gib`. Chọn máy thuê với RAM dư lớn so với ước lượng và có dung lượng ổ đĩa cho dữ liệu, mô hình cùng kết quả. Đặc biệt theo dõi Random Forest trên 17,6 triệu dòng và XGBoost dense. Nếu full mode báo thiếu RAM, thay đổi cấu hình hoặc dùng máy lớn hơn; chương trình không âm thầm lấy mẫu.

## Kết quả và giới hạn đánh giá

Mỗi lần chạy có mã riêng dưới `modelTraining/outputs/` (hoặc `--output-dir`): `models/<model>/<run_id>/` chứa mô hình và `run_manifest.json`, `metrics/<run_id>/<model>/` chứa `metrics.csv` và `attack_type_results.csv`, `logs/<run_id>.log` chứa nhật ký. Khi chạy cả ba, `metrics/<run_id>/model_comparison.csv` gom các dòng chỉ số. Thư mục đầu ra mặc định nằm trong `.gitignore`.

Các chỉ số gồm Accuracy, Precision, Recall, F1, Macro-F1, PR-AUC, Average Precision, TN/FP/FN/TP, tỷ lệ báo nhầm và thời gian dự đoán. Các hàng `attack_type_results.csv` là recall cho từng loại attack trong bài toán nhị phân. Smoke test chỉ đánh giá mẫu validation; full mode đánh giá toàn bộ validation rồi test sau khi cấu hình và ngưỡng đã cố định. Validation chứa Password mới và rất nhiều attack, nên không nên coi nó là đại diện cho mọi kiểu tấn công. Đừng chọn mô hình hoặc chỉnh tham số theo test rồi công bố test là đánh giá độc lập.

`rss_after_fit_gib` là lượng RAM của tiến trình đo sau fit, **không phải đỉnh RAM**. Việc đo và dự báo bộ nhớ là gần đúng. Đối chiếu log và mức sử dụng RAM thực tế trên máy thuê trước khi chạy lâu.
