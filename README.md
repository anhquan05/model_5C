# 🏦 Dự báo Rủi ro Tín dụng Khách hàng - Mô hình 5C

Ứng dụng Streamlit chuyển đổi từ notebook huấn luyện mô hình **Logistic Regression**
(`4_7_qtrr.ipynb`) để dự báo xác suất rủi ro tín dụng (`PD`: 0 = không rủi ro,
1 = có rủi ro) dựa trên 24 biến khảo sát thuộc 5 nhóm tiêu chí tín dụng cổ điển
(**5C**):

| Nhóm | Ý nghĩa | Các biến |
|---|---|---|
| TC | Tư cách (Character) | TC1–TC5 |
| NL | Năng lực (Capacity) | NL1–NL4 |
| DK | Điều kiện (Condition) | DK1–DK5 |
| V  | Vốn (Capital) | V1–V6 |
| TS | Tài sản đảm bảo (Collateral) | TS1–TS4 |

Mô hình sử dụng đúng `LogisticRegression` của scikit-learn như trong notebook gốc,
với phép chia tập train/test `test_size=0.2` và `random_state=32` (đều có thể
tùy chỉnh trên giao diện).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

Tệp CSV cần chứa ít nhất các cột sau:
- 24 biến đầu vào (thang điểm Likert 1–5): `TC1..TC5`, `NL1..NL4`, `DK1..DK5`,
  `V1..V6`, `TS1..TS4`.
- Cột nhãn `PD` (0 hoặc 1) — biến mục tiêu (biến rủi ro tín dụng).

Các cột khác trong tệp gốc (`Dấu thời gian`, `NN`) không được dùng làm biến
đầu vào của mô hình (đúng theo notebook gốc) và sẽ bị bỏ qua khi huấn luyện.

## Mô tả các tab

1. **⚙️ Sidebar (Cấu hình & Tải dữ liệu)** — tải tệp CSV, cấu hình tỷ lệ tập
   kiểm tra, random state, và các tham số nâng cao của Logistic Regression
   (C, max_iter, solver). Nút **"Huấn luyện mô hình"** là nơi duy nhất kích hoạt
   quá trình huấn luyện.
2. **📋 Tổng quan dữ liệu** — kích thước dữ liệu, xem nhanh dữ liệu thô, thống
   kê mô tả của 24 biến đầu vào và biến mục tiêu.
3. **📊 Trực quan hóa dữ liệu** — phân phối lớp rủi ro (PD) và biểu đồ phân
   phối của tối đa 4 biến đầu vào do người dùng chọn (do có tới 24 biến).
4. **🧪 Kết quả huấn luyện & kiểm định mô hình** — Accuracy, Precision, Recall,
   F1-score, ROC-AUC, ma trận nhầm lẫn, đường cong ROC và classification report
   chi tiết. Chỉ hiển thị sau khi đã bấm nút huấn luyện.
5. **🔮 Sử dụng mô hình** — dự báo cho một khách hàng bằng cách nhập trực tiếp
   điểm khảo sát theo từng nhóm 5C, hoặc dự báo hàng loạt bằng cách tải lên tệp
   CSV có đúng 24 cột biến đầu vào (có nút tải kết quả CSV sau khi dự báo).

## Ghi chú kỹ thuật

- Notebook gốc **không** áp dụng bước chuẩn hóa/scale dữ liệu (do các biến đều
  ở cùng thang điểm Likert 1–5), nên ứng dụng cũng không thêm scaler để giữ
  đúng pipeline gốc.
- Notebook gốc không tinh chỉnh siêu tham số của `LogisticRegression` (dùng mặc
  định của scikit-learn: `C=1.0`, `max_iter=100`, `solver="lbfgs"`) — các giá
  trị này được giữ làm mặc định trên giao diện, người dùng có thể điều chỉnh
  trong mục "Tham số nâng cao".
- Khuyến nghị dùng Streamlit phiên bản mới (≥1.38) để đảm bảo tương thích với
  các thành phần layout (`st.container(height=...)`, `st.tabs`, `st.form`).
