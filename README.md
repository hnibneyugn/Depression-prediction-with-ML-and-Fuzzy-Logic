# 🧠 Công cụ Sàng lọc Sức khỏe Tinh thần Sinh viên

Ứng dụng hỗ trợ sinh viên tự đánh giá sớm các yếu tố rủi ro liên quan đến trầm cảm, kết hợp mô hình **Machine Learning (XGBoost)** và hệ thống **Luật Mờ (Fuzzy Logic)** để đưa ra kết quả đa chiều và đáng tin cậy.

> ⚠️ **Lưu ý quan trọng:** Công cụ này **không phải chẩn đoán y tế**. Kết quả chỉ mang tính **tham khảo** và không thay thế ý kiến của chuyên gia tâm lý.

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Pipeline xử lý](#pipeline-xử-lý)
- [Hệ thống Luật Mờ](#hệ-thống-luật-mờ)
- [Đánh giá mô hình](#đánh-giá-mô-hình)
- [Bảo mật dữ liệu](#bảo-mật-dữ-liệu)

---

## Tổng quan

Ứng dụng được xây dựng dựa trên bộ dữ liệu **Student Depression Dataset** (27.901 mẫu, 18 thuộc tính) với mục tiêu sàng lọc ban đầu nguy cơ trầm cảm ở sinh viên. Hệ thống kết hợp hai phương pháp chính:

1. **XGBoost** — mô hình Machine Learning dự đoán xác suất trầm cảm dựa trên thông tin đầu vào
2. **Fuzzy Logic (scikit-fuzzy)** — hệ thống luật mờ 5 đầu vào tổng hợp đa chiều để đưa ra đánh giá cuối cùng

Ứng dụng đặc biệt chú trọng **nhạy cảm lâm sàng**: mức "Rất Cao" chỉ dành cho trường hợp có ý tự tử, các trường hợp khác được giới hạn ở mức "Cao".

---

## Kiến trúc hệ thống

```
Người dùng nhập thông tin
        │
        ▼
┌───────────────────────┐
│  Bước 1: Nhập liệu   │  Học tập, Tài chính, Giấc ngủ, Tự tử, v.v.
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Bước 2: XGBoost      │  predict_proba → xác suất ML (vd: 0.55)
│  (Routing theo        │  • Có ý tự tử → xgb_model.pkl
│   suy nghĩ tự tử)    │  • Không → xgb_model_ns.pkl
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Bước 3: Fuzzy Logic  │  5 đầu vào: academic_pressure, financial_stress,
│  (scikit-fuzzy)       │  suicidal_thoughts, sleep_duration, ml_probability
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Bước 4: Kết luận     │  Điểm rủi ro 0-100 → Phân loại mức độ
│  cuối cùng            │  Rất Thấp / Thấp / Trung Bình / Cao / Rất Cao
└───────────────────────┘
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|---|---|---|
| Giao diện | Streamlit | ≥ 1.30.0 |
| ML Model | XGBoost | ≥ 2.0.0 |
| Fuzzy Logic | scikit-fuzzy | ≥ 0.5.0 |
| Xử lý dữ liệu | Pandas, NumPy | ≥ 2.0.0, ≥ 1.24.0 |
| ML Framework | scikit-learn | ≥ 1.3.0 |
| Giải thích mô hình | SHAP | ≥ 0.43.0 |
| Serialization | joblib | ≥ 1.3.0 |
| Đồ thị | Matplotlib | ≥ 3.7.0 |
| Khác | SciPy, NetworkX | ≥ 1.10.0, ≥ 3.0 |

---

## Cài đặt

### Yêu cầu hệ thống
- Python 3.11+
- pip

### Các bước cài đặt

```bash
# 1. Clone repository
git clone <repository-url>
cd "Depression Prediction"

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Chạy ứng dụng
streamlit run app.py
```

### Sử dụng Dev Containers (VS Code)
Dự án hỗ trợ **Dev Containers** với Python 3.11. Mở project trong VS Code và chọn "Reopen in Container" để tự động cấu hình môi trường phát triển.

---

## Hướng dẫn sử dụng

### 1. Nhập thông tin

Người dùng điền các thông tin:

**Thông tin cá nhân:**
- Giới tính
- Tuổi (15-60)
- Nghề nghiệp / Ngành
- Điểm GPA (0-10)

**Yếu tố rủi ro chính:**
- Mức áp lực học tập (1-5)
- Mức áp lực tài chính (1-5)
- Thời gian ngủ trung bình
- Suy nghĩ tự tử (Có/Không)

**Thông tin bổ sung (tùy chọn):**
- Áp lực công việc
- Mức hài lòng học tập/công việc
- Giờ học/làm việc mỗi ngày
- Chế độ ăn

### 2. Kết quả đánh giá

Sau khi nhấn **"Bắt đầu sàng lọc"**, hệ thống sẽ hiển thị:
- **Kết luận cuối cùng** với mức rủi ro và điểm đánh giá (0-100)
- **Chi tiết quy trình 4 bước** đánh giá
- **Giải thích ngôn ngữ tự nhiên** về lý do đánh giá
- **Thông tin hỗ trợ** (đường dây tư vấn tâm lý nếu cần)

### Phân loại mức rủi ro

| Điểm | Mức độ | Ý nghĩa |
|---|---|---|
| ≥ 85 | Rất cao | Có nguy cơ trầm cảm CAO |
| ≥ 65 | Cao | Có nguy cơ trầm cảm |
| ≥ 30 | Trung bình | Có một số yếu tố cần theo dõi |
| < 30 | Thấp | Nguy cơ trầm cảm thấp |

---

## Cấu trúc dự án

```
Depression Prediction/
├── app.py                 # Ứng dụng Streamlit chính
├── Analyst.ipynb          # Notebook phân tích & huấn luyện mô hình
├── xgb_model.pkl          # Mô hình XGBoost (có yếu tố tự tử)
├── xgb_model_ns.pkl       # Mô hình XGBoost (không có yếu tố tự tử)
├── threshold.pkl          # Ngưỡng phân loại tối ưu
├── requirements.txt       # Dependencies
├── .devcontainer/         # Cấu hình Dev Container
│   └── devcontainer.json
└── README.md              # Tài liệu hướng dẫn
```

---

## Pipeline xử lý

### Xử lý dữ liệu (Analyst.ipynb)

1. **Đọc dữ liệu** — Student Depression Dataset (27.901 mẫu)
2. **Tiền xử lý:**
   - Loại bỏ cột không cần thiết (`id`, `Degree`, `City`)
   - Chuyển đổi `Sleep Duration` từ text sang số
   - Mã hóa nhị phân: `Gender`, `Suicidal Thoughts`, `Family History`
   - Mã hóa thứ tự: `Dietary Habits`
   - One-hot encoding: `Profession`
3. **Huấn luyện mô hình:**
   - Logistic Regression
   - Random Forest
   - **XGBoost** (được chọn — hiệu suất tốt nhất)
   - XGBoost phiên bản không có yếu tố tự tử

### Routing thông minh

Hệ thống sử dụng **2 mô hình XGBoost riêng biệt**:
- `xgb_model.pkl` — khi người dùng có suy nghĩ tự tử
- `xgb_model_ns.pkl` — khi không có suy nghĩ tự tử (không sử dụng feature này)

Chiến lược này giúp tăng độ chính xác cho từng nhóm đối tượng.

---

## Hệ thống Luật Mờ

### 5 Đầu vào (Antecedents)

| Biến | Khoảng giá trị | Membership Functions |
|---|---|---|
| Áp lực học tập | 1-5 | Thấp, Trung bình, Cao |
| Áp lực tài chính | 1-5 | Thấp, Trung bình, Cao |
| Suy nghĩ tự tử | 0-1 | Không, Có |
| Thời lượng giấc ngủ | 3-10 giờ | Thiếu ngủ, Bình thường, Đủ giấc |
| Xác suất ML (XGBoost) | 0-1 | Thấp, Lấp lửng, Cao |

### Đầu ra (Consequent)

- **Depression Risk**: 0-100
- 5 mức: Rất thấp, Thấp, Trung bình, Cao, Rất cao

### Nhóm luật chính

- **Nhóm A — Có suy nghĩ tự tử:** Luôn nghiêm trọng, phân biệt theo áp lực
- **Nhóm B — Không tự tử, ML cao:** Giấc ngủ đủ giúp giảm mức, nhưng cap ở "cao" (không bao giờ "rất cao" nếu không có ý tự tử)
- **Nhóm C — Không tự tử, ML lấp lửng:** Tương tự nhóm B nhưng mức thấp hơn
- **Nhóm D — Không tự tử, ML thấp:** Nghiêng về an toàn, thiếu ngủ vẫn đẩy lên 1 bậc

### Cơ chế "Sleep Protection"

Giấc ngủ đủ (≥ 7.5 giờ) đóng vai trò yếu tố bảo vệ, có thể giảm mức rủi ro xuống 1 bậc — **trừ khi** cả áp lực học tập và tài chính đều ở mức tối đa.

---

## Đánh giá mô hình

### So sánh các mô hình (trên tập test)

| Mô hình | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.857 | 0.883 | 0.918 |
| LR (no suicidal) | 0.799 | 0.811 | 0.858 | 0.869 |
| Random Forest | 0.837 | 0.849 | 0.878 | 0.911 |
| **XGBoost** | **0.847** | **0.857** | **0.886** | **0.918** |
| XGBoost (no suicidal) | 0.798 | 0.808 | 0.860 | 0.870 |

### Tối ưu ngưỡng (Threshold)

Ngưỡng tối ưu dựa trên F1-score: **0.5** (F1 = 0.871)

---

## Bảo mật dữ liệu

🔒 **Cam kết bảo mật:**
- Toàn bộ dữ liệu chỉ xử lý **tạm thời trong bộ nhớ**
- **Không lưu trữ** bất kỳ thông tin người dùng nào
- **Không gửi đến máy chủ bên ngoài**
- Xử lý hoàn toàn phía client (trình duyệt)

---

## Liên hệ hỗ trợ

Nếu bạn hoặc người thân đang gặp khó khăn về sức khỏe tinh thần:
- 📞 **Tổng đài tư vấn tâm lý:** 1900 1267 (miễn phí, 24/7)
- 💬 Hoặc nói chuyện với một người bạn tin tưởng, thầy cô, hoặc người thân
