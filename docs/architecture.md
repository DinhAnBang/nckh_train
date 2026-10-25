# Cấu trúc mã nguồn

Mã nguồn chính nằm trong `src/ton_iot` và được chia theo trách nhiệm:

- `config`: đường dẫn dự án và cấu hình mô hình.
- `data`: sáu bước đọc, phân tích, xác minh, làm sạch, chia tập và xử lý đặc trưng.
- `training`: đọc dữ liệu đặc trưng, đánh giá và huấn luyện ba mô hình.
- `validation`: kiểm tra độc lập dữ liệu đầu ra.
- `reporting`: tạo các báo cáo nghiên cứu.
- `pipeline`: backup và điều phối pipeline.

Các file mã nguồn cũ trong `dataPreprocessing`, `modelTraining` và
`pipeline_tools` đã được xóa sau khi chuyển logic sang `src`. Dataset, output
và báo cáo không nằm trong cây mã nguồn này và không bị di chuyển.

Từ thư mục gốc, có thể cài package ở chế độ editable:

```powershell
python -m pip install -e .
python -m ton_iot.training.train --model all --dry-run
```

Nếu chưa cài editable, có thể đặt `src` vào `PYTHONPATH` khi chạy lệnh.
