# Phân tích kết quả LightGBM trên ToN-IoT Network (1 triệu dòng)

## Phạm vi và nguồn kiểm tra

Báo cáo này phân tích lần chạy `20261025T121329Z_a016b31f`. Chỉ đọc dữ liệu, mã nguồn và kết quả hiện có; không sửa dữ liệu, không chia lại tập, không huấn luyện lại mô hình. Các số liệu bên dưới thuộc **mẫu 1 triệu dòng**, không phải kết quả đánh giá trên toàn bộ 22,3 triệu dòng.

- Kết quả Colab: [cấu hình và thông tin lần chạy](../notebooks/lightgbm/models/lightgbm/20261025T121329Z_a016b31f/run_manifest.json), [chỉ số đánh giá](../notebooks/lightgbm/metrics/20261025T121329Z_a016b31f/lightgbm/metrics.csv), [kết quả theo loại tấn công](../notebooks/lightgbm/metrics/20261025T121329Z_a016b31f/lightgbm/attack_type_results.csv), [nhật ký](../notebooks/lightgbm/logs/20261025T121329Z_a016b31f.log) và `model.joblib` cùng thư mục với cấu hình.
- Dữ liệu đối chiếu: [báo cáo lấy mẫu](../dataPreprocessing/outputs/experiments/network_1m/sampling_report.json), [báo cáo xác minh mẫu](../dataPreprocessing/outputs/experiments/network_1m/validation_report.json), [quy tắc tạo đặc trưng](../dataPreprocessing/outputs/experiments/network_1m/metadata/train_fitted_rules.json) và các tệp Parquet `train`, `validation`, `test` trong cùng thư mục `network_1m`.
- Mã nguồn đã đọc: [làm sạch](../src/ton_iot/data/cleaner.py), [chia tập](../src/ton_iot/data/splitter.py), [tạo đặc trưng](../src/ton_iot/data/preprocessor.py), [lấy mẫu](../src/ton_iot/training/experiments.py), [huấn luyện](../src/ton_iot/training/train.py), [nạp dữ liệu](../src/ton_iot/training/data_loader.py), [đánh giá](../src/ton_iot/training/evaluate.py). Quy tắc được học từ **toàn bộ 17.599.039 dòng train gốc**, rồi dùng để tạo cùng 197 cột cho ba tập; sau đó mới lấy mẫu 800.000/100.000/100.000 dòng để thí nghiệm.

Tệp `model.joblib` có SHA-256 `e3baae2bdcc230af927518129ba573bc9ab482d5e35246a0387e54f749210deb`. Đây là dấu vân tay của bản tải về, **không phải xác nhận tệp đáng tin cậy**. Định dạng này chứa mã tuần tự hóa Python, có thể thực thi mã khi `joblib.load`; môi trường hiện tại còn khác phiên bản Colab (LightGBM 4.7.0 so với 4.6.0; scikit-learn 1.9.1 so với 1.6.1). Vì vậy, tôi **không gọi `joblib.load`**. Tôi kiểm tra cấu trúc tệp, lấy phần văn bản mô hình LightGBM bên trong rồi dùng `lightgbm.Booster` của môi trường hiện tại để dự đoán theo từng phần 20.000 dòng. Kết quả tái tạo **đúng cả bốn ô** của ma trận nhầm lẫn validation và test trong `metrics.csv`. Xác suất và độ quan trọng đặc trưng nêu dưới đây được tính từ phần mô hình đó; chúng không phải tệp xác suất do Colab lưu riêng, và khả năng có sai khác rất nhỏ giữa hai phiên bản thư viện chưa được kiểm chứng từng dòng. Thứ tự 197 cột trong Parquet khớp chính xác với cấu hình Colab và quy tắc xử lý.

Hai mã SHA-256 ghi trong cấu hình Colab cho báo cáo lấy mẫu và quy tắc tạo đặc trưng đã được đối chiếu với các tệp hiện tại:

| Tệp | SHA-256 đối chiếu, trùng cấu hình Colab |
|---|---|
| `sampling_report.json` | `3a553b17ce2e3561b95c2a367ea12fc63449dbd85fcec9574f57c68eb5b0782f` |
| `train_fitted_rules.json` | `d127ce258a1c5ade0229f49af3950084f71617b4e89e38ea01866faf9dc199cb` |

## LightGBM đã học được gì và đang sai ở đâu?

Mô hình được huấn luyện trên 800.000 dòng, 197 đặc trưng, dùng ngưỡng xác suất tấn công 0,5, không đặt trọng số lớp; thời gian huấn luyện ghi nhận 10,605 giây. Phần mô hình đọc được chứa 122 cây; cấu hình đặt giới hạn 400 cây và dừng sớm theo validation. Mô hình nhận diện tốt nhiều mẫu tấn công, nhưng việc nhận diện bình thường giảm mạnh khi chuyển sang test.

| Tập | Bình thường / tấn công | Bình thường nhận đúng | Bình thường báo nhầm | Tấn công bỏ sót | Tấn công nhận đúng | Tỷ lệ báo nhầm trong lớp bình thường |
|---|---:|---:|---:|---:|---:|---:|
| Train, kiểm tra lại trên mẫu dùng học | 21.178 / 778.822 | 20.864 | 314 | 164 | 778.658 | 1,48% |
| Validation | 2.102 / 97.898 | 2.081 | 21 | 36 | 97.862 | 1,00% |
| Test | 9.812 / 90.188 | 4.422 | **5.390** | 328 | 89.860 | **54,93%** |

Trên test, accuracy 94,282%, F1 lớp bình thường 0,607334 và F1 lớp tấn công 0,969165. Mô hình đoán tấn công cho **95.250/100.000** dòng test (89.860 đúng + 5.390 sai), trong khi nhãn tấn công thực là 90.188. Cách đoán mọi dòng đều là tấn công chỉ đạt accuracy 90,188% và báo nhầm toàn bộ 9.812 dòng bình thường. Vì vậy LightGBM có học được tín hiệu hữu ích, nhưng chi phí báo động sai trên test vẫn quá cao. Accuracy chung che khuất vấn đề này do tấn công chiếm đa số.

Theo tệp kết quả theo loại, test có 17.030 backdoor (bỏ sót 6), 35 MITM (bỏ sót 6), 2.440 ransomware (bỏ sót 11) và 70.683 XSS (bỏ sót 305). Validation chỉ có password trong lớp tấn công: 97.898 dòng, bỏ sót 36. Mẫu MITM rất nhỏ nên không nên suy rộng tỷ lệ của loại này.

## Bằng chứng về thay đổi dữ liệu bình thường

### Cách chia và độ lệch nhãn

[Bộ chia](../src/ton_iot/data/splitter.py) đặt scanning/DoS/injection/DDoS ở train, password ở validation, XSS/ransomware/backdoor/MITM ở test; các tập cũng được xếp theo thời gian. Mẫu 1 triệu dòng lấy độc lập trong từng tập theo loại và 10 khoảng thời gian, không trộn thứ tự thời gian. Train mẫu có 2,647% bình thường, validation 2,102%, test 9,812%. Trong toàn bộ dữ liệu trước lấy mẫu, tỷ lệ bình thường lần lượt là 465.883/17.599.039, 36.879/1.755.447 và 292.749/2.983.666; vậy sự thay đổi tỷ lệ lớp ở test đã tồn tại trong nguồn, không phải do mẫu được tạo ra.

Mẫu train trải từ ngày 02/04 đến 26/04/2019 UTC, validation từ 26/04 đến 27/04, test từ 27/04 đến 29/04. Việc giữ riêng các đợt tấn công và thời gian khiến validation và test đo **những tình huống khác nhau**. Validation đạt tốt không bảo đảm test cũng tốt. Tuy nhiên, chỉ từ kết quả này không thể tách tác động riêng của thời gian, loại tấn công và thay đổi lưu lượng bình thường: các yếu tố diễn ra cùng lúc. Loại tấn công mới không trực tiếp giải thích tỷ lệ báo nhầm vì tỷ lệ ấy chỉ tính trên dòng có nhãn bình thường.

### Cảnh báo sai tập trung ở đâu?

| Nhóm dòng bình thường của test | Số dòng | Báo nhầm | Tỷ lệ báo nhầm |
|---|---:|---:|---:|
| `Network_dataset_20.csv` | 354 | 9 | 2,54% |
| `Network_dataset_21.csv` | 489 | 4 | 0,82% |
| `Network_dataset_22.csv` | 7.847 | **5.077** | **64,70%** |
| `Network_dataset_23.csv` | 1.122 | 300 | 26,74% |
| TCP | 6.781 | 5.369 | 79,18% |
| UDP | 2.907 | 18 | 0,62% |

Ngày 28/04/2019 UTC có 5.076 báo nhầm trên 7.725 dòng bình thường (65,71%); ngày 27/04 chỉ 15/1.061 (1,41%). `source_file` và ngày **không nằm trong 197 đặc trưng**; chúng giúp định vị giai đoạn phát sinh lỗi, không chứng minh rằng tên tệp hay thời gian tự gây lỗi.

Một mẫu đáng chú ý là **TCP, `service='-'`, `src_bytes=0`, `dst_bytes=0`**:

| Tập | Bình thường có mẫu này | Tấn công có mẫu này | Bình thường có mẫu này bị báo nhầm |
|---|---:|---:|---:|
| Train | 9.717/21.178 | 682.827/778.822 | 113/9.717 |
| Validation | 4/2.102 | 24.427/97.898 | 4/4; mẫu quá nhỏ để ước lượng tỷ lệ |
| Test | **5.340/9.812** | 20.764/90.188 | **5.063/5.340** |

Như vậy **5.063/5.390 (93,93%)** cảnh báo sai thuộc mẫu trên. Phần còn lại có 327 cảnh báo sai trong 4.472 dòng bình thường không thuộc mẫu này (7,31%). Mẫu không hoàn toàn mới với train, nhưng ở train nó xuất hiện trong rất nhiều dòng tấn công; phân bố và các đặc trưng đi kèm ở test đã khác. Đây là **bằng chứng mạnh về mối liên hệ giữa kiểu lưu lượng và lỗi**. Chưa thể khẳng định riêng bốn điều kiện trên là nguyên nhân đủ: mô hình dùng nhiều đặc trưng đồng thời. Cần so sánh có kiểm soát trong tập phát triển để xác định vai trò từng đặc trưng.

Trạng thái kết nối `OTH` cũng đáng chú ý: 4.867/5.390 cảnh báo sai test có trạng thái này, so với 123/4.422 dòng bình thường được nhận đúng. Trong test có 4.990 dòng bình thường `OTH`, nên tỷ lệ báo nhầm trong nhóm này là 97,54%. Ở validation không có dòng bình thường nào đồng thời mang mẫu TCP/`-`/hai trường byte bằng 0/**`OTH`**; ở test có 4.863 dòng như vậy. Đây là một khoảng trống quan trọng của validation cho tình huống xảy ra nhiều ở test.

### Xác suất dự đoán và đặc trưng quan trọng

Xác suất tấn công trung vị của **toàn bộ dòng bình thường** là 0,00485 ở train, 0,00836 ở validation và **0,92306 ở test**. Với riêng 5.390 cảnh báo sai test, trung vị là **0,95435**; 5.052 dòng có xác suất ít nhất 0,8, 4.098 dòng ít nhất 0,95, 1.151 dòng ít nhất 0,99. Đây không chỉ là nhiều dòng nằm sát ngưỡng 0,5. Mô hình thường **rất tự tin nhưng sai** trên nhóm bình thường mới này; chỉ tăng nhẹ ngưỡng khó giải quyết tận gốc, và mọi thử ngưỡng phải dùng dữ liệu phát triển, không chọn theo test.

Độ quan trọng theo tổng mức cải thiện khi chia cây, đọc từ mô hình đã lưu:

| Đặc trưng | Tỷ lệ trong tổng mức cải thiện |
|---|---:|
| `src_pkts` | 24,46% |
| `src_ip_bytes` | 13,87% |
| `proto=tcp` | 13,30% |
| `dst_port` | 8,50% |
| `dst_pkts` | 6,08% |
| `dns_rejected=T` | 5,97% |
| `dns_rejected=F` | 5,90% |

Tên mã hóa đã được đối chiếu với từ điển train: `proto__cat_001=tcp`, `dns_rejected__cat_002=T`, `dns_rejected__cat_001=F`. Đây là mức sử dụng trong cây, **không phải bằng chứng tác động nhân quả**. Ví dụ, trung vị `src_ip_bytes` của bình thường train/validation/test lần lượt là 67/344/76, nhưng riêng bình thường báo nhầm test là 52 so với 129 ở nhóm được nhận đúng. `src_bytes` của hai nhóm test có trung vị 0 so với 41; `dst_bytes` trung bình 16,5 so với 2.271,5. Phân bố bình thường test thay đổi theo các đại lượng mà mô hình dùng nhiều. Cổng đích cũng có vai trò lớn; một số cổng trong test có tỷ lệ báo nhầm cao, nhưng tên cổng đơn lẻ có thể chỉ phản ánh môi trường của đợt ghi dữ liệu.

## Kiểm tra sai nhãn, trùng lặp và xử lý đặc trưng

- **Xử lý nhất quán: đạt ở mức cấu trúc.** Mã nguồn học điền thiếu và từ điển mã hóa chỉ từ train, dùng lại cho validation/test. Báo cáo kiểm tra mẫu xác nhận 197 cột cùng thứ tự, `X/y/metadata` thẳng hàng, không thiếu hoặc vô hạn trong `X`, `record_id` duy nhất, thứ tự thời gian đúng. Trong mẫu hiện tại, tổng số ô cờ `__unknown` bằng 0 ở cả ba tập; sự bùng nổ cảnh báo sai không thể quy ngay cho danh mục chưa thấy. Hash quy tắc và báo cáo mẫu trùng với cấu hình Colab. Điều này kiểm tra dữ liệu đầu vào và quy tắc, chưa chứng minh từng giá trị gốc đều đúng về mặt ngữ nghĩa.
- **Trùng theo nguồn: không thấy giao nhau giữa các tập trong nhóm bình thường đã kiểm tra.** `duplicate_key` được tạo từ 46 cột gốc, khác với việc hai dòng có cùng 197 đặc trưng. Nhóm bình thường có 11 dòng lặp khóa trong train, 0 trong validation, 1.580 trong test; giữa các tập không có khóa trùng ở nhóm này. [Kiểm tra chia tập](../dataPreprocessing/reports/splitting/leakage_check.csv) cũng ghi nhận không giao nhau theo khóa nguồn trên tập đầy đủ. Không xóa những dòng trùng này.
- **Trùng đặc trưng: có và cần phân biệt.** [Báo cáo xác minh mẫu](../dataPreprocessing/outputs/experiments/network_1m/validation_report.json) ghi 112 nhóm, 540 dòng có cùng 197 giá trị đặc trưng ở hơn một tập. So khớp lại bằng hai hàm băm đặc trưng như mã nguồn: trong 5.390 cảnh báo sai test chỉ **3 dòng** có bộ đặc trưng trùng với train, cả 3 trùng với dòng train mang nhãn tấn công; 4.422 dòng bình thường nhận đúng có **131 dòng** trùng train, đều trùng nhãn bình thường. Trong chính nhóm 5.390 cảnh báo sai có 2.890 dòng lặp lại bộ đặc trưng đã xuất hiện trong nhóm đó. Trùng đặc trưng có thể do bỏ IP/thời gian hoặc do lưu lượng lặp lại, không tự chứng minh trùng bản ghi gốc, rò rỉ hay sai nhãn. Vì chỉ 3 cảnh báo sai trùng X với train, trùng X xuyên tập không giải thích phần lớn 5.390 lỗi.
- **Sai nhãn ngữ nghĩa: chưa xác minh.** Chương trình chỉ xác nhận `label=0` đi cùng `type=normal` và `label=1` đi cùng loại tấn công. Nó không thể xác nhận độc lập liệu lưu lượng TCP `OTH` của ngày 28/04 thực sự vô hại. Cần xem lại một số `record_id` gốc, nhật ký thu thập và định nghĩa nhãn cùng người am hiểu dữ liệu trước khi sửa nhãn. Ba trường hợp cùng X với train tấn công là tín hiệu để xem xét, không đủ kết luận sai nhãn.

## Kết luận theo mức độ chắc chắn

| Nhận định | Mức độ | Căn cứ / giới hạn |
|---|---|---|
| Mô hình thiên về nhãn tấn công trên test | **Đạt, đã xác minh** | Dự đoán tấn công 95.250 dòng so với 90.188 nhãn thực; 54,93% bình thường bị báo nhầm. |
| Phân bố bình thường của test khác train và validation, tập trung ở TCP/`OTH`/hai trường byte bằng 0 | **Đạt, đã xác minh** | Đếm trực tiếp 5.063 cảnh báo sai trong một mẫu TCP; nhóm này gần như không có trong bình thường validation. |
| Sự thay đổi phân bố góp phần tạo ra lỗi | **Có bằng chứng mạnh, chưa chứng minh nhân quả đầy đủ** | Lỗi tập trung cùng sự thay đổi mẫu; cần phép thử đối chứng trên dữ liệu phát triển để tách vai trò từng yếu tố. |
| Chia theo thời gian và loại tấn công làm validation không đại diện cho test | **Đạt về tính khác biệt; ảnh hưởng riêng chưa xác minh** | Quy tắc chia và đếm dữ liệu xác nhận các giai đoạn khác nhau; thời gian và loại tấn công bị gắn với nhau. |
| Mất cân bằng lớp góp phần khiến accuracy gây hiểu lầm | **Đạt** | Train chỉ 2,647% bình thường, test 9,812%; baseline luôn đoán tấn công đạt 90,188% accuracy. Mức đóng góp chính xác của mất cân bằng vào 5.390 lỗi chưa xác minh. |
| Sai nhãn gốc, lỗi mã hóa hoặc rò rỉ là nguyên nhân chính | **Chưa xác minh / chưa có bằng chứng ủng hộ** | Kiểm tra cấu trúc và đối chiếu mô hình đều khớp; không có kiểm định độc lập về ngữ nghĩa nhãn. |

Vì vậy, vấn đề thấy rõ nhất nằm ở **sự thay đổi lưu lượng bình thường giữa các giai đoạn** và phản ứng của mô hình dùng đặc trưng từng kết nối trước thay đổi ấy. Không đủ bằng chứng để quy hết lỗi cho thuật toán LightGBM, cho sai nhãn hoặc cho một cột duy nhất. Giao thức chia hiện tại vẫn có giá trị khi mục tiêu nghiên cứu là kiểm tra khả năng tổng quát hóa theo thời gian và loại tấn công mới; không nên đổi giao thức chỉ để làm chỉ số test đẹp hơn.

## Thứ tự thí nghiệm nên thực hiện tiếp

1. **Rà soát nhãn và nguồn của một nhóm đại diện, không sửa dữ liệu ngay.** Lấy `record_id` của các dòng test bình thường báo nhầm, đặc biệt từ `Network_dataset_22.csv` và trạng thái `OTH`; so với dòng bình thường nhận đúng, tệp CSV gốc, mô tả đợt thu thập và quy tắc gán nhãn. Ghi lại bằng chứng cho từng trường hợp. Nếu tìm thấy nhãn sai, thiết kế lại bộ đánh giá dưới phiên bản dữ liệu mới và giữ test cũ như mốc lịch sử.
2. **Tạo một tập phát triển bổ sung theo thời gian từ dữ liệu train/validation hiện có hoặc từ một đợt thu thập mới, giữ nguyên test hiện tại.** Tập này cần có đủ lưu lượng bình thường TCP/`OTH` giống tình huống triển khai; đo tỷ lệ báo nhầm riêng theo ngày, tệp, giao thức, trạng thái và cổng. Nếu nhóm đó vẫn lỗi, giả thuyết lệch phân bố được củng cố. Không dùng test đã phân tích để chọn mô hình, ngưỡng hay cột đặc trưng rồi tiếp tục gọi nó là test độc lập; nếu cần lựa chọn theo tình huống mới, phải dành một test tương lai khác.
3. **Trên tập phát triển, thử các phép loại bỏ từng nhóm đặc trưng và điều chỉnh mất cân bằng.** Thử riêng cổng đích, `conn_state`, giao thức, byte/gói và các biến DNS; so sánh LightGBM với cấu hình hiện tại, theo tỷ lệ báo nhầm của bình thường và tỷ lệ phát hiện tấn công ở cùng ngưỡng. Thử trọng số lớp hoặc ngưỡng chọn bằng validation phù hợp. Đây là thí nghiệm mới, chưa thực hiện trong báo cáo này; điều chỉnh ngưỡng đơn thuần có thể không đủ vì 4.098 cảnh báo sai có xác suất từ 0,95 trở lên.
4. **Sau khi có mốc đánh giá công bằng, mới thử đặc trưng ngữ cảnh theo cửa sổ thời gian.** Mô hình hiện dùng thông tin của từng kết nối; nó không thấy tần suất kết nối, số đích khác nhau, chuỗi thất bại/thành công hay lịch sử ngắn của một nguồn. Cửa sổ trượt theo IP nguồn/đích và thời gian có thể giúp phân biệt một kết nối đơn lẻ giống tấn công với hành vi lặp lại, nhưng **chưa có bằng chứng rằng nó sẽ sửa 5.390 báo nhầm**. Tạo các đặc trưng chỉ từ sự kiện đã xảy ra trước thời điểm dự đoán, xử lý ranh giới tập rõ ràng, tránh dùng thông tin tương lai, rồi so sánh với mô hình từng kết nối trên cùng một tập đánh giá mới.

Ưu tiên trước mắt là **kiểm tra ngữ nghĩa nhãn của cụm TCP/`OTH` ngày 28/04 và xây dựng tập phát triển có lưu lượng bình thường tương tự**. Cửa sổ trượt là hướng thử sau, không phải điều kiện bắt buộc để tiếp tục nghiên cứu.
