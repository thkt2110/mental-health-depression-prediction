# 🧠 Kế Hoạch Triển Khai: Exploring Mental Health Data
## Kaggle Playground Series S4E11

**Môn học**: CSC17001 — Intelligent Data Analytics  
**Thời gian**: 7/4 → 23/4/2026 (16 ngày)  
**Nhóm**: Thành (Leader), Đàm Đạt, Tuấn, Kiệt, Tâm

---

## 📜 Quy Tắc Chung

### Code & Notebook
1. Mỗi biểu đồ/bảng **BẮT BUỘC** kèm ô Markdown phân tích bên dưới: giải thích insights, nhận xét xu hướng, rút ra kết luận. Không được để biểu đồ trơn không giải thích.
2. Dữ liệu gốc trong `data/raw/` **KHÔNG ĐƯỢC chỉnh sửa**. Output lưu vào `data/processed/`.
3. Đảm bảo train và test được xử lý **cùng pipeline** (tránh data leakage).
4. Trước khi nộp, mỗi người **tự chạy lại** notebook mình phụ trách: `Kernel → Restart & Run All`.

### Git & Commit
5. Commit message theo chuẩn **Conventional Commits**: `<type>(<scope>): <mô tả ngắn>`
   - **Type thường dùng**: `feat`, `fix`, `chore`, `docs`, `refactor`, `style`, `test`
   - **Scope** (tuỳ chọn): `eda`, `prep`, `fe`, `model`, `eval`, `report`
   - Ví dụ:
     ```
     feat(eda): add target distribution analysis
     feat(prep): handle missing values for student vs professional
     feat(model): train XGBoost with 5-fold CV
     docs(report): write feature engineering section
     fix(fe): fix data leakage in target encoding
     chore: update requirements.txt
     ```
6. Mỗi lần hoàn thành 1 phần việc có ý nghĩa → commit + push, đừng gom lại commit 1 cục lớn.

### Report (LaTeX)
7. **Ai làm phần nào thì viết Report phần đó** vào template LaTeX. Không đẩy hết cho 1 người.

### Giao tiếp
9. Cập nhật tiến độ hàng ngày trên nhóm chat.
10. Nếu thấy không kịp deadline → **báo sớm ít nhất 2 ngày** để nhóm điều chỉnh.

---

## 📅 Tổng Quan 5 Giai Đoạn

| GĐ | Thời gian | Mục tiêu |
|---|---|---|
| **1** | 7/4 → 11/4 | Hiểu bài toán + Hoàn thành EDA |
| **2** | 10/4 → 15/4 | Preprocessing + Feature Engineering |
| **3** | 14/4 → 19/4 | Training Baseline + Advanced Models |
| **4** | 18/4 → 21/4 | Evaluation + Kaggle Submission |
| **5** | 20/4 → 23/4 | Finalize Report + Video + Nộp bài |

```
GĐ1  ║██████████║                                       7/4 – 11/4
GĐ2        ║███████████║                                10/4 – 15/4
GĐ3                 ║███████████║                       14/4 – 19/4
GĐ4                          ║████████║                18/4 – 21/4
GĐ5                                ║████████║          20/4 – 23/4
```

---

## 🔶 GĐ1 — EDA (7/4 → 11/4)

**Mục tiêu**: Toàn nhóm nắm rõ bài toán + `01_eda.ipynb` hoàn chỉnh

| Thành viên | Công việc |
|---|---|
| **Thành** | EDA: phân tích bivariate (từng feature vs Depression), correlation heatmap |
| **Tuấn** | EDA: phân phối target, univariate analysis cho categorical + numerical features |
| **Kiệt** | Nghiên cứu public notebooks trên Kaggle → tổng hợp cách xử lý data hay, chia sẻ nhóm |
| **Đàm Đạt** | Nghiên cứu public notebooks trên Kaggle → tổng hợp modeling strategies, chia sẻ nhóm |
| **Tâm** | Setup LaTeX Report template (đúng 10 sections yêu cầu) + viết phần Project Overview |

**Deliverable**: `01_eda.ipynb` hoàn chỉnh (Thành + Tuấn viết phần Data Exploration trong Report)

---

## 🔶 GĐ2 — Preprocessing & Feature Engineering (10/4 → 15/4)

**Mục tiêu**: Data sạch + features chất lượng, sẵn sàng cho modeling

| Thành viên | Công việc |
|---|---|
| **Kiệt** | `02_preprocessing.ipynb`: xử lý missing values, encoding categorical, merge conditional columns + viết Report phần Preprocessing |
| **Đàm Đạt** | `03_feature_engineering.ipynb`: Target Encoding (City, Profession, Degree), interaction features, feature selection + viết Report phần FE |
| **Tâm** | `03_feature_engineering.ipynb`: cùng Đạt triển khai — age binning, risk score, correlation-based selection + viết Report phần FE |
| **Thành** | Review pipeline end-to-end: kiểm tra tính nhất quán train/test, review cả NB02 + NB03 |
| **Tuấn** | Viết Report phần Data Exploration + vẽ visualizations cho features mới từ NB03 |

**Deliverable**: `02_preprocessing.ipynb` + `03_feature_engineering.ipynb` hoàn chỉnh, `data/processed/` có data sẵn sàng

---

## 🔶 GĐ3 — Modeling (14/4 → 19/4)

**Mục tiêu**: Train đa dạng models, tìm ra best single model + ensemble

| Thành viên | Công việc |
|---|---|
| **Tuấn** | `04_model_baseline.ipynb`: Logistic Regression + Decision Tree (5-Fold Stratified CV) |
| **Kiệt** | `04_model_baseline.ipynb`: Random Forest + SVM (5-Fold Stratified CV) + viết Report phần Model Development (Baseline) |
| **Đàm Đạt** | `05_model_advanced.ipynb`: XGBoost + CatBoost (hyperparameter tuning) |
| **Tâm** | `05_model_advanced.ipynb`: LightGBM (hyperparameter tuning) |
| **Thành** | `05_model_advanced.ipynb`: Ensemble (weighted average / stacking top models) + viết Report phần Model Development (Advanced) |

**Deliverable**: `04_model_baseline.ipynb` + `05_model_advanced.ipynb` hoàn chỉnh, biết được best model + ensemble strategy

---

## 🔶 GĐ4 — Evaluation & Submission (18/4 → 21/4)

**Mục tiêu**: Đánh giá tổng thể + submit Kaggle + hoàn thiện nội dung Report

| Thành viên | Công việc |
|---|---|
| **Đàm Đạt** | `06_evaluation.ipynb`: tổng hợp kết quả, Confusion Matrix, Classification Report + generate `submission.csv` |
| **Tuấn** | `06_evaluation.ipynb`: SHAP / Feature Importance, ROC-AUC curves, bảng so sánh models |
| **Thành** | Submit Kaggle (late submission) + screenshot kết quả + viết Report phần Kaggle Results |
| **Kiệt** | Viết Report phần Experiments & Results (bảng so sánh, phân tích kết quả) |
| **Tâm** | Viết Report phần Conclusion & Future Work + chuẩn bị slides cho video |

**Deliverable**: `06_evaluation.ipynb` hoàn chỉnh, `submission.csv` đã submit, Report gần xong

---

## 🔶 GĐ5 — Report, Video & Nộp Bài (20/4 → 23/4)

**Mục tiêu**: Finalize mọi thứ + nộp đúng hạn

| Thành viên | Công việc |
|---|---|
| **Thành** | Re-run ALL notebooks, review + finalize Report, đóng gói ZIP nộp bài |
| **Đàm Đạt** | Review nội dung technical trong Report, thử submit thêm các version khác nếu cần |
| **Tuấn** | Kiểm tra toàn bộ visualizations trong notebooks + Report, sửa nếu cần |
| **Kiệt** | Hoàn thiện phần Task Distribution + References trong Report |
| **Tâm** | Tổng hợp format Report PDF + edit video + upload YouTube |
| **ALL** | Quay video: mỗi người trình bày phần mình |

**Phân đoạn Video (12-15 phút):**

| Thứ tự | Người | Nội dung | ~Thời lượng |
|---|---|---|---|
| 1 | **Thành** | Project Overview + Pipeline tổng thể | 2-3 phút |
| 2 | **Tuấn** | Data Exploration + Key Insights | 2-3 phút |
| 3 | **Kiệt** | Preprocessing + Baseline Models | 2-3 phút |
| 4 | **Đàm Đạt** | Feature Engineering + Advanced Models | 2-3 phút |
| 5 | **Tâm** | Evaluation + Kaggle Score + Conclusion | 2-3 phút |

**Deliverable**: ZIP file nộp bài + video YouTube

---

## 📊 Tổng Hợp Phân Công

### Notebooks

| Notebook | Phụ trách | Deadline |
|---|---|---|
| `01_eda.ipynb` | Thành + Tuấn | 11/4 |
| `02_preprocessing.ipynb` | Kiệt | 13/4 |
| `03_feature_engineering.ipynb` | Đàm Đạt + Tâm | 15/4 |
| `04_model_baseline.ipynb` | Tuấn + Kiệt | 17/4 |
| `05_model_advanced.ipynb` | Đàm Đạt + Tâm + Thành | 19/4 |
| `06_evaluation.ipynb` | Đàm Đạt + Tuấn | 21/4 |

### Report (LaTeX) — Ai làm phần nào viết phần đó

| Section | Người viết |
|---|---|
| Project Planning & Task Distribution | Kiệt |
| Project Overview | Tâm |
| Data Exploration | Thành + Tuấn |
| Data Preprocessing | Kiệt |
| Feature Engineering | Đàm Đạt + Tâm |
| Model Development | Kiệt (Baseline) + Thành (Advanced) |
| Experiments & Results | Kiệt |
| Conclusion & Future Work | Tâm |
| Kaggle Screenshot | Thành |
| References | Kiệt |

### Tỷ lệ đóng góp đề xuất

| Thành viên | % |
|---|---|
| Thành | 20% |
| Đàm Đạt | 20% |
| Tuấn | 20% |
| Kiệt | 20% |
| Tâm | 20% |

---

## 🎯 Checklist Trước Khi Nộp (23/4)

- [ ] 6 notebooks chạy thành công (`Restart & Run All`), mọi biểu đồ có markdown phân tích
- [ ] Report PDF đủ 10 sections, có screenshot Kaggle + YouTube link
- [ ] Video YouTube: cả 5 TV trình bày, giọng thật
- [ ] `submission.csv` đã submit Kaggle (93,800 rows, format `id,Depression`)
- [ ] `TeamName.zip` nộp đúng hạn (< 25MB hoặc Google Drive)
