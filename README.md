# StudentDSS – Hệ thống Hỗ trợ Quyết định Cảnh báo Sớm và Khuyến nghị Học tập Sinh viên

**StudentDSS** là hệ thống hỗ trợ quyết định (Decision Support System) giúp nhận diện sớm sinh viên có nguy cơ học vụ ngay sau Học kỳ 1 và đề xuất kế hoạch can thiệp cá nhân hóa cho Cố vấn học tập. Hệ thống kết hợp kiến trúc phân loại học máy hai giai đoạn (**Random Forest + SMOTE + XGBoost**), trí tuệ nhân tạo khả giải (**SHAP Analysis**) và bộ máy khuyến nghị lọc theo nội dung (**Content-Based Filtering - CBF**) trên không gian ngữ nghĩa 6 chiều.

---

## Tính năng nổi bật

* **Báo cáo Tổng quan (Batch Prediction):** Hỗ trợ tải lên danh sách sinh viên định dạng CSV (như `data_demo.csv`) để dự đoán hàng loạt và thống kê tỷ lệ phân bổ 3 nhóm trạng thái: `Safe` (An toàn), `Caution` (Cảnh báo) và `Dropout` (Nguy cơ bỏ học).


* **Phân tích Cá nhân & Giải thích Mô hình (XAI):** Tích hợp biểu đồ `SHAP Analysis` đo lường mức độ tác động trung bình của từng đặc trưng (`mean(|SHAP value|)`) lên kết quả đầu ra của mô hình đối với từng sinh viên cụ thể.


* **Hệ thống Khuyến nghị Học tập (CBF Engine):** Tự động đối chiếu hồ sơ học thuật của sinh viên với danh mục giải pháp (`recommendation_items.csv`) để gợi ý các hành động can thiệp phù hợp kèm mức độ tương đồng và các yếu tố kích hoạt cụ thể.


* **Mô phỏng & Nhập liệu Thủ công:** Cho phép nhập trực tiếp các chỉ số học tập Học kỳ 1 trên giao diện web để kiểm tra nhanh kết quả phân loại và nhận diện hướng xử lý.



---

## Kiến trúc Hệ thống & Phương pháp luận

### 1. Dữ liệu & Kỹ thuật Đặc trưng (Feature Engineering)

Hệ thống sử dụng tập dữ liệu kết quả học tập Học kỳ 1 đã qua tiền xử lý (`dataset_preprocessed.csv`) gồm 7 đặc trưng gốc và 1 cột nhãn mục tiêu (`Target`):

* `1st_sem_enrolled`: Số môn học/tín chỉ đăng ký.


* `1st_sem_evaluations`: Số môn học tham gia đánh giá.


* `1st_sem_without_evaluations`: Số môn học không tham gia đánh giá (bỏ thi).


* `1st_sem_approved`: Số môn học đạt (qua môn).


* `1st_sem_failed`: Số môn học không đạt (rớt môn).


* `1st_sem_grade`: Điểm trung bình Học kỳ 1.


* `1st_sem_pass_rate`: Tỷ lệ qua môn Học kỳ 1.



Từ 7 đặc trưng gốc, hệ thống tính toán thêm **2 đặc trưng phái sinh** để nâng cao khả năng phân loại (tổng cộng 9 đặc trưng đầu vào cho mô hình học máy):

* `failed_burden` (Gánh nặng rớt môn): Được tính bằng `1st_sem_failed / 1st_sem_enrolled`.


* `ghosting_rate` (Tỷ lệ bỏ thi): Được tính bằng `1st_sem_without_evaluations / 1st_sem_enrolled`.



### 2. Mô hình Phân loại Hai giai đoạn (Two-Stage Classification)

Để giải quyết bài toán mất cân bằng dữ liệu và hạn chế cảnh báo giả, quy trình dự báo được chia thành 2 bước tuần tự:

* **Giai đoạn 1 – Sàng lọc sơ bộ (`Safe` vs. `At-Risk`):** Sử dụng thuật toán **Random Forest** (`n_estimators=300`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`) để tách nhóm sinh viên an toàn (`Safe`) ra khỏi nhóm có dấu hiệu rủi ro (`At-Risk`) dựa trên cơ chế bỏ phiếu đa số (Majority Voting: $\hat{Y} = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\}$).


* **Giai đoạn 2 – Phân loại chuyên sâu (`Caution` vs. `Dropout`):** Các sinh viên thuộc diện `At-Risk` được đưa qua kỹ thuật cân bằng mẫu **SMOTE** (`random_state=42`) trước khi huấn luyện bằng mô hình **XGBoost** (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `eval_metric='logloss'`).


* **Tinh chỉnh ngưỡng quyết định (Threshold Tuning):** Áp dụng ngưỡng xác suất `caution_threshold = 0.45` (nếu xác suất lớp `Caution` $\ge 0.45$ thì gán nhãn `Caution`, ngược lại gán nhãn `Dropout`).



### 3. Bộ máy Khuyến nghị Lọc nội dung (6D Semantic CBF)

Mô-đun `cbf.py` ánh xạ thông số học thuật của sinh viên vào **Không gian ngữ nghĩa 6 chiều**:

* `low_grade`: Mức độ thiếu hụt điểm số (`max_grade - 1st_sem_grade`).


* `low_pass_rate`: Mức độ thiếu hụt tỷ lệ qua môn (`1 - 1st_sem_pass_rate`).


* `failed_burden`: Áp lực học lại từ các môn rớt.


* `evaluation_issue`: Mức độ bỏ thi dựa trên `ghosting_rate`.


* `workload_gap`: Chênh lệch giữa khối lượng đăng ký và số môn thực đạt (`(1st_sem_enrolled - 1st_sem_approved) / 1st_sem_enrolled`).


* `high_performance`: Chỉ số phản ánh năng lực học tập tốt (`1st_sem_grade * 1st_sem_pass_rate`).



Hệ thống sử dụng độ tương đồng Cosine (**Cosine Similarity**) để đo khoảng cách giữa Vector Sinh viên và các Vector Giải pháp trong `recommendation_items.csv`, từ đó xếp hạng và trích xuất các hành động hỗ trợ cụ thể.

---

## Cấu trúc Thư mục Dự án

```text
StudentDSS/
├── dataset/                          # Thư mục chứa dữ liệu
│   ├── data_demo.csv                 # Dữ liệu mẫu dùng để kiểm thử trên giao diện
│   ├── dataset.csv                   # Tập dữ liệu gốc ban đầu
│   └── dataset_preprocessed.csv      # Tập dữ liệu đã qua tiền xử lý
├── docs/                             # Tài liệu kỹ thuật của dự án
│   └── pipeline.md                   # Mô tả chi tiết luồng xử lý dữ liệu và mô hình
├── models/                           # Các artifacts mô hình đã huấn luyện
│   ├── feature_columns.pkl           # Danh sách 9 đặc trưng đầu vào của mô hình ML
│   ├── rf_model1.pkl                 # Mô hình Random Forest (Giai đoạn 1)
│   ├── scaler.pkl                    # Bộ chuẩn hóa MinMaxScaler 6 chiều ngữ nghĩa
│   └── xgb_model2.pkl                # Mô hình XGBoost (Giai đoạn 2)
├── recommendation/                   # Module Hệ thống Khuyến nghị
│   ├── cbf.py                        # Mã nguồn xử lý thuật toán Content-Based Filtering
│   └── recommendation_items.csv      # Danh mục giải pháp và trọng số vector chuyên gia
├── .gitignore                        # Cấu hình các file bỏ qua của Git
├── app.py                            # Ứng dụng giao diện người dùng Streamlit
├── build_cbf_engine.ipynb            # Notebook xây dựng và kiểm thử bộ máy khuyến nghị CBF
├── data_preprocessing.ipynb          # Notebook làm sạch và tiền xử lý dữ liệu
├── requirements.txt                  # Danh sách các thư viện Python cần thiết
└── train_model.ipynb                 # Notebook huấn luyện mô hình RF, SMOTE, XGBoost

```
---

## Kết quả Thực nghiệm & Đánh giá

Tập dữ liệu gồm 4.424 sinh viên được chia theo tỷ lệ 80% Train (3.539 mẫu) và 20% Test (885 mẫu) có phân tầng (`stratify=y`, `random_state=42`). Hiệu năng của mô hình trên tập Test được đo lường qua các chỉ số `Accuracy`, `Precision`, `Recall` và `F1-Score`:

| Lớp (Class) | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| --- | --- | --- | --- | --- |
| **Safe (An toàn)** | 0.77 | 0.80 | 0.79 | 442 |
| **Dropout (Bỏ học)** | 0.72 | 0.57 | 0.63 | 284 |
| **Caution (Cảnh báo)** | 0.34 | 0.42 | 0.38 | 159 |
| **Accuracy tổng thể** |  |  | **0.66** | **885** |
| **Macro Average** | **0.61** | **0.60** | **0.60** | **885** |
| **Weighted Average** | **0.67** | **0.66** | **0.66** | **885** |

---
## Hướng dẫn Cài đặt & Khởi chạy

**Bước 1: Tải mã nguồn, khởi tạo và kích hoạt môi trường ảo**
Sau khi clone kho lưu trữ về máy, hãy di chuyển vào thư mục gốc `StudentDSS`, tạo môi trường ảo mới và kích hoạt:

```bash
git clone https://github.com/NguyenHuann/StudentDSS.git
cd StudentDSS

# 1. Khởi tạo môi trường ảo (.venv)
python -m venv .venv

# 2. Kích hoạt môi trường ảo:
# - Trên Windows (CMD / PowerShell):
.venv\Scripts\activate
# - Trên Linux / macOS:
source .venv/bin/activate

```

**Bước 2: Cài đặt các thư viện phụ thuộc**
Cài đặt toàn bộ gói thư viện thông qua file `requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`, `joblib`, `streamlit`):

```bash
pip install -r requirements.txt

```

**Bước 3: Khởi chạy ứng dụng Web**
Chạy file `app.py` để mở giao diện tương tác trên trình duyệt:

```bash
streamlit run app.py

```

---

## Quy trình Tái huấn luyện (Pipeline)

Nếu muốn chạy lại toàn bộ quy trình từ dữ liệu thô đến khi sinh ra mô hình, hãy thực thi lần lượt các Jupyter Notebook theo thứ tự sau:

1. `data_preprocessing.ipynb`: Tiền xử lý dữ liệu từ `dataset/dataset.csv` và xuất ra `dataset/dataset_preprocessed.csv`.


2. `train_model.ipynb`: Trích xuất đặc trưng, huấn luyện mô hình 2 giai đoạn và lưu các file `.pkl` vào thư mục `models/`.


3. `build_cbf_engine.ipynb`: Thiết lập danh mục khuyến nghị `recommendation/recommendation_items.csv` và kiểm thử logic trong `recommendation/cbf.py`.