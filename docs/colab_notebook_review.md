# Rà soát notebook Colab cho bộ mẫu Network 1 triệu dòng

Ngày kiểm tra: 25/10/2026. Phạm vi: mã nguồn và tệp trên máy Windows; **chưa chạy notebook trên Google Colab**, chưa huấn luyện lại mô hình.

## A. Những gì đã sửa

| Tệp | Thay đổi và lý do |
|---|---|
| `notebooks/colab_network_1m.ipynb` | Kiểm tra đúng repository, nhánh `refactor/unified-src` và commit; chỉ cập nhật Git kiểu nhanh tiến khi mã nguồn sạch. Dùng `PYTHONPATH` và kiểm tra import thay cho bước cài editable bắt buộc. Hỗ trợ thư mục hoặc ZIP trên Drive; kiểm tra đủ tệp và dừng nếu xác minh bộ mẫu thất bại. Huấn luyện một mô hình khi người dùng bật `RUN_TRAIN=True`; XGBoost GPU cần bài kiểm tra CUDA đạt. Đọc manifest/metrics theo đúng mã lần chạy, hiển thị kết quả, tạo biểu đồ từ số liệu đã lưu. Các ô không chứa kết quả thực thi hoặc token. |
| `.gitignore` | Bổ sung mẫu dữ liệu, mô hình, ZIP, kết quả, backup, cache, môi trường và tệp bí mật thường gặp; giữ notebook, `src/`, tài liệu và cấu hình để đưa lên Git. |
| `docs/colab_1m_experiment.md` | Sửa hướng dẫn cũ còn yêu cầu cài editable; cập nhật quy trình thư mục/ZIP và đọc kết quả. |
| `docs/colab_notebook_review.md` | Báo cáo kiểm tra này. |

## B. Kiểm tra mã nguồn

- **Đạt tại máy local:** Notebook là JSON hợp lệ; cả 13 ô Python biên dịch được; không có kết quả hoặc token lưu trong ô. Import `ton_iot`, `python -m ton_iot.training.experiments --help` và `python -m ton_iot.training.train --help` chạy được bằng `PYTHONPATH=src`. Các cờ notebook dùng (`validate`, `--gpu-check`, `--train-mode full`, `--experiment network_1m`, `--device`, `--max-ram-gib`...) đều có trong mã nguồn. Kiểm tra bố cục Parquet hiện có: train 800.000, validation 100.000, test 100.000 dòng; 197 đặc trưng.
- **Đạt ở mức kiểm thử logic:** Với dữ liệu kết quả LightGBM thật, notebook đọc đúng `run_manifest.json`, `metrics.csv`, `attack_type_results.csv` và nhận được 5.390 cảnh báo sai trên test. Tạo được 5 biểu đồ PNG trong bộ nhớ, không ghi kết quả kiểm thử vào dự án. Trường hợp mô hình thiếu hoặc nhiều mã lần chạy được xử lý rõ. Tên tệp ZIP dạng gốc và dạng bọc một thư mục được kiểm tra trong bộ nhớ; đường dẫn ZIP nguy hiểm bị từ chối. Bộ mẫu có sẵn được dùng lại, bộ mẫu thiếu bị chặn. Kiểm tra Git bẩn, sai origin và thứ tự lệnh cập nhật bằng mô phỏng; không cập nhật Git thật.
- **Không cần sửa `src/`:** `train.py` đã có kiểm tra RAM trước khi nạp toàn bộ train, bài fit XGBoost nhỏ để kiểm tra CUDA, lưu kết quả dưới `models/`, `metrics/`, `logs/`, và ghi thời gian/RAM/GPU vào manifest. Mã tạo đặc trưng học quy tắc từ train; notebook chỉ đọc bộ mẫu hiện có. `pyproject.toml` có cấu hình đóng gói hợp lệ. Nguyên nhân chính xác khiến `pip install -e` từng thất bại trên Colab chưa xác minh vì không có nhật ký lỗi đó; notebook hiện không phụ thuộc vào bước này.
- **Chưa xác minh trên Colab:** cài thư viện, truy cập Drive/GitHub, giải nén ZIP và bài fit CUDA trên Tesla T4; không thể khẳng định các bước này đã chạy thành công từ kiểm thử Windows. [Thông báo chính thức của Colab](https://github.com/googlecolab/colabtools/issues/6081) cho biết runtime mặc định đã chuyển sang Python 3.13 từ tháng 8/2026; dự án yêu cầu Python từ 3.10 và các phiên bản thư viện trong `requirements.txt` chưa được cài thử trên runtime Colab hiện tại. Notebook in phiên bản Python và dừng ngay nếu cài hoặc import thất bại.

## C. Kiểm tra `.gitignore`

- Dữ liệu gốc và đầu ra `dataPreprocessing/outputs/`, báo cáo sinh tự động, CSV, Parquet, ZIP, `.joblib`, các thư mục `results/`, `colab_data/`, backup, cache Python và môi trường ảo đều bị loại khỏi Git. Kết quả tải về trong `notebooks/lightgbm/` cũng bị loại.
- Các mẫu `.env`, khóa và tệp thông tin đăng nhập thường gặp bị loại. Không có CSV, Parquet hoặc mô hình nào đang được Git theo dõi trong 36 tệp đã theo dõi.
- `src/`, notebook `.ipynb`, `requirements.txt`, `pyproject.toml` và tài liệu Markdown vẫn có thể được theo dõi. `git check-ignore -v` xác nhận dữ liệu và mô hình bị bỏ qua, còn notebook và tài liệu không bị bỏ qua.

## D. Trạng thái Git trước khi push

- Nhánh hiện tại: `refactor/unified-src`, đang trùng với `origin/refactor/unified-src` tại thời điểm kiểm tra; commit hiện tại `266b935d5abe51b378dfb80f37a3f5840d601bbe`.
- Tệp thay đổi: `.gitignore`, `notebooks/colab_network_1m.ipynb`, `docs/colab_1m_experiment.md`. Tệp mới: báo cáo này và `docs/lightgbm_1m_analysis.md` từ lần phân tích trước. Cần quyết định có đưa báo cáo LightGBM lên cùng commit hay không.
- `git diff --check` đạt; chỉ có thông báo Git sẽ chuẩn hóa LF/CRLF trên Windows. Không thấy chuỗi khóa riêng, token GitHub/AWS hoặc mật khẩu gắn cứng trong các tệp nguồn và tài liệu dự kiến đưa lên. Notebook không lưu output. Tệp thay đổi lớn nhất là notebook, khoảng 32 KB; không có dữ liệu lớn trong danh sách dự kiến.
- `git add --dry-run .` cho thấy các tệp mã nguồn/tài liệu trên và **không có** Parquet, CSV, ZIP hoặc mô hình. Chưa thực hiện `git add`, commit hay push.

## E. Vấn đề còn tồn tại

Notebook cần được chạy thật một lần trên Colab để xác nhận tải mã nguồn, cài thư viện, Drive và GPU T4. Nếu có nhiều kết quả của cùng mô hình, người dùng phải điền mã lần chạy cụ thể; notebook không tự chọn tệp cuối cùng. Hai mô hình XGBoost và Random Forest chưa có kết quả tải về nên biểu đồ hiện chỉ dùng LightGBM thật và báo thiếu hai mô hình kia.

Tỷ lệ cảnh báo sai cao của LightGBM trên test là **hạn chế của mô hình/dữ liệu nghiên cứu**, không phải lỗi đường dẫn notebook. Phân tích chi tiết nằm trong `docs/lightgbm_1m_analysis.md`; lần sửa này không đổi nhãn, đặc trưng, chia tập, ngưỡng hay tham số.

## F. Kết luận trước khi push

**SẴN SÀNG PUSH mã nguồn và tài liệu sau khi bạn xem lại danh sách tệp.** Kiểm tra tại máy local không phát hiện dữ liệu lớn hoặc bí mật trong các tệp dự kiến đưa lên. Kết luận này chỉ về trạng thái Git và mã notebook; thành công trên runtime Colab hiện tại vẫn phải kiểm chứng khi chạy ở Colab.
