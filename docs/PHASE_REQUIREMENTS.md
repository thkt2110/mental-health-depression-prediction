# 📋 Yêu Cầu Chi Tiết Từng Giai Đoạn

Tài liệu này mô tả **sườn yêu cầu** cho từng notebook / giai đoạn. Mỗi thành viên follow theo sườn này khi thực hiện phần việc của mình.

> **Nhắc lại**: Mỗi biểu đồ / bảng kết quả **BẮT BUỘC** kèm ô Markdown phân tích bên dưới.

---

## 🔶 GĐ1 — `01_eda.ipynb` (Thành + Tuấn)

### Mục tiêu
Hiểu rõ dữ liệu, tìm patterns, phát hiện vấn đề cần xử lý ở GĐ2.

### Cấu trúc notebook

#### 1.1 Load Data & Tổng quan
- Load `train.csv` và `test.csv`
- In `.shape`, `.dtypes`, `.head()`, `.info()`
- Liệt kê bảng tổng quan tất cả cột: tên, kiểu dữ liệu, số missing, số unique values
- **Markdown**: Nhận xét tổng quan về kích thước data, kiểu dữ liệu chính, tỷ lệ missing

#### 1.2 Phân tích Target Variable
- Countplot + tỷ lệ % giữa Depression = 0 vs 1
- **Markdown**: Data có bị imbalanced không? Mức độ ra sao? Có cần xử lý imbalance không?

#### 1.3 Phân tích Missing Values
- Heatmap hoặc bar chart hiển thị % missing mỗi cột
- Phân tích pattern: so sánh missing giữa Student vs Working Professional
  - `Academic Pressure`, `CGPA`, `Study Satisfaction` → chỉ có ở Student?
  - `Work Pressure`, `Job Satisfaction` → chỉ có ở Working Professional?
- **Markdown**: Giải thích tại sao missing, đề xuất hướng xử lý (fill/merge/drop)

#### 1.4 Univariate Analysis — Numerical Features
Với mỗi cột numerical (`Age`, `Academic Pressure`, `Work Pressure`, `CGPA`, `Study Satisfaction`, `Job Satisfaction`, `Work/Study Hours`, `Financial Stress`):
- Histogram + KDE plot
- `.describe()` (mean, std, min, max, quartiles)
- Boxplot để phát hiện outliers
- **Markdown cho mỗi feature**: phân phối có lệch không? Có outliers đáng chú ý không?

#### 1.5 Univariate Analysis — Categorical Features
Với mỗi cột categorical (`Gender`, `City`, `Working Professional or Student`, `Profession`, `Sleep Duration`, `Dietary Habits`, `Degree`, `Have you ever had suicidal thoughts ?`, `Family History of Mental Illness`):
- Countplot (top 15-20 nếu high cardinality)
- Bảng frequency table
- **Markdown cho mỗi feature**: phân phối đều hay lệch? Có quá nhiều categories không? Có categories hiếm cần gom nhóm?

#### 1.6 Bivariate Analysis — Features vs Depression
Với mỗi feature, phân tích mối quan hệ với target:
- **Numerical**: boxplot hoặc violin plot grouped by Depression, hoặc KDE plot riêng 2 lớp
- **Categorical**: stacked bar chart hoặc grouped countplot theo Depression
- **Markdown cho mỗi feature**: Feature này có khả năng phân biệt Depression cao hay thấp? Giải thích pattern nhìn thấy.

#### 1.7 Correlation Analysis
- Correlation heatmap cho numerical features (include Depression)
- Top features tương quan mạnh nhất với Depression
- **Markdown**: Nhận xét cặp features nào tương quan cao (multicollinearity?), features nào tương quan mạnh với target

#### 1.8 Key Findings & Đề xuất
- **Markdown tổng kết**:
  - Liệt kê 5-7 key findings quan trọng nhất
  - Đề xuất cụ thể cho GĐ2: cần xử lý gì ở preprocessing, features nào nên tạo/loại bỏ

---

## 🔶 GĐ2a — `02_preprocessing.ipynb` (Kiệt)

### Mục tiêu
Làm sạch dữ liệu, encoding, tạo data sẵn sàng cho Feature Engineering.

### Cấu trúc notebook

#### 2.1 Load Data & Recap
- Load `train.csv` và `test.csv`
- Tóm tắt ngắn lại key findings từ EDA liên quan đến preprocessing

#### 2.2 Drop Cột Không Cần Thiết
- Drop `id` (lưu lại riêng cho submission), `Name` (không có ý nghĩa dự đoán)
- **Markdown**: Giải thích lý do drop

#### 2.3 Xử Lý Missing Values
Đây là phần quan trọng nhất vì data có **conditional missing**:
- `Academic Pressure`, `CGPA`, `Study Satisfaction` → NaN nếu Working Professional
- `Work Pressure`, `Job Satisfaction` → NaN nếu Student

Hướng xử lý đề xuất (nhóm tự quyết định):
- **Cách 1 (Merge)**: Tạo cột chung `Pressure` = `Academic Pressure` nếu Student, `Work Pressure` nếu Professional. Tương tự `Satisfaction` = `Study Satisfaction` / `Job Satisfaction`.
- **Cách 2 (Fill 0/median)**: Fill NaN bằng 0 hoặc median theo group.
- `CGPA`: Fill NaN (Working Professional) bằng 0 hoặc median.

Cần in ra: số missing trước và sau xử lý mỗi cột, đảm bảo hết NaN.

- **Markdown**: Giải thích strategy đã chọn và tại sao

#### 2.4 Encoding Categorical Variables
- **Binary features** → 0/1:
  - `Gender` (Male=1, Female=0 hoặc ngược lại)
  - `Have you ever had suicidal thoughts ?` (Yes=1, No=0)
  - `Family History of Mental Illness` (Yes=1, No=0)
  - `Working Professional or Student` (Working Professional=1, Student=0)
- **Ordinal features** → mapping thứ tự:
  - `Sleep Duration`: "Less than 5 hours" < "5-6 hours" < "7-8 hours" < "More than 8 hours" (cần xác nhận lại categories thực tế)
  - `Dietary Habits`: "Unhealthy" < "Moderate" < "Healthy"
- **High cardinality** (`City`, `Profession`, `Degree`) → **để lại cho notebook FE** xử lý bằng Target Encoding

In ra `.dtypes` và `.head()` sau encoding.

- **Markdown**: Giải thích mapping cho mỗi cột, lý do chọn loại encoding

#### 2.5 Xử Lý Outliers (nếu cần)
- Dựa trên phát hiện từ EDA
- Chọn strategy: clip (cap), remove, hoặc giữ nguyên
- **Markdown**: Giải thích quyết định

#### 2.6 Kiểm Tra & Lưu Data
- In `.info()`, `.describe()`, kiểm tra không còn NaN
- Đảm bảo **train và test được xử lý bằng cùng logic**
- Lưu output ra `data/processed/train_preprocessed.csv` và `test_preprocessed.csv`

---

## 🔶 GĐ2b — `03_feature_engineering.ipynb` (Đàm Đạt + Tâm)

### Mục tiêu
Tạo features mới, encoding features phức tạp, chọn features tốt nhất cho modeling.

### Cấu trúc notebook

#### 3.1 Load Preprocessed Data
- Load từ `data/processed/train_preprocessed.csv` + `test_preprocessed.csv`

#### 3.2 Target Encoding cho High Cardinality Features
Với `City`, `Profession`, `Degree`:
- Sử dụng Target Encoding (dùng `category_encoders.TargetEncoder` hoặc tự implement)
- **Quan trọng**: phải dùng **cross-validation** khi fit target encoding trên train để tránh data leakage
- In bảng: top 10 / bottom 10 giá trị encoded cho mỗi cột
- **Markdown**: Giải thích Target Encoding là gì, tại sao dùng cho high cardinality, cách tránh leakage

#### 3.3 Tạo Interaction Features
Đề xuất (nhóm chọn lọc):
- `Pressure × Financial_Stress` → kết hợp 2 nguồn stress
- `Age × Work_Study_Hours` → tải công việc theo tuổi
- `Risk_Score` = `Suicidal_Thoughts` + `Family_History` → yếu tố rủi ro kép
- Hoặc các interactions khác nhóm nghĩ ra từ EDA

Với mỗi feature mới:
- Vẽ phân phối
- Vẽ relationship với Depression
- **Markdown**: Feature này có ý nghĩa gì? Có giúp phân biệt Depression tốt hơn không?

#### 3.4 Binning (nếu phù hợp)
- `Age` → age groups (e.g., 18-25, 26-35, 36-45, 46+)
- `Work/Study Hours` → nhóm thời gian
- **Markdown**: Tại sao chọn binning? Bins có hợp lý không?

#### 3.5 Feature Selection
- Tính correlation của tất cả features với Depression
- Dùng feature importance từ Random Forest hoặc Mutual Information
- Vẽ bar chart top features
- Loại bỏ features dư thừa / multicollinear nếu cần
- **Markdown**: Kết luận giữ lại bao nhiêu features, liệt kê danh sách final

#### 3.6 Lưu Data Final
- Lưu `data/processed/train_final.csv` + `test_final.csv`
- In `.shape`, `.columns`, `.head()` của data final
- **Markdown**: Tóm tắt: bắt đầu với bao nhiêu features → kết thúc bao nhiêu → liệt kê

---

## 🔶 GĐ3a — `04_model_baseline.ipynb` (Tuấn + Kiệt)

### Mục tiêu
Train các model cơ bản làm baseline, thiết lập benchmark accuracy.

### Cấu trúc notebook

#### 4.1 Load Data & Setup
- Load `data/processed/train_final.csv`
- Tách `X_train`, `y_train`
- Setup `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Định nghĩa hàm evaluate chung: train → predict → tính accuracy, f1, precision, recall trên mỗi fold → trả về mean ± std

#### 4.2 Logistic Regression
- `StandardScaler` + `LogisticRegression`
- 5-Fold CV
- In kết quả: accuracy mỗi fold + mean ± std
- **Markdown**: Nhận xét accuracy. Model tuyến tính có phù hợp với bài toán này không?

#### 4.3 Decision Tree
- `DecisionTreeClassifier` với vài giá trị `max_depth` (5, 10, 15, None)
- 5-Fold CV cho mỗi config
- In kết quả so sánh max_depth
- **Markdown**: max_depth nào tốt nhất? Có dấu hiệu overfit không?

#### 4.4 Random Forest
- `RandomForestClassifier` với vài giá trị `n_estimators` (100, 200, 500)
- 5-Fold CV
- In kết quả + feature importance plot
- **Markdown**: So sánh với Decision Tree. Ensemble có cải thiện không?

#### 4.5 SVM
- `StandardScaler` + `SVC(kernel='linear')` hoặc `SVC(kernel='rbf')`
- 5-Fold CV (nếu data lớn quá, dùng subset ~20-30k rows)
- **Markdown**: Performance so với tree-based models?

#### 4.6 Bảng So Sánh Baseline
- Bảng tổng hợp:

| Model | Accuracy (mean ± std) | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ... | ... | ... | ... |
| Decision Tree (best) | ... | ... | ... | ... |
| Random Forest (best) | ... | ... | ... | ... |
| SVM | ... | ... | ... | ... |

- **Markdown**: Nhận xét tổng hợp: model nào tốt nhất? Tree-based vs linear? Baseline accuracy đạt bao nhiêu? Expectations cho advanced models?

---

## 🔶 GĐ3b — `05_model_advanced.ipynb` (Đàm Đạt + Tâm + Thành)

### Mục tiêu
Train GBDT models + ensemble để đạt accuracy cao nhất cho Kaggle submission.

### Cấu trúc notebook

#### 5.1 Load Data & Setup
- Load `data/processed/train_final.csv`
- Tách X, y
- Setup `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (cùng seed với baseline)

#### 5.2 XGBoost (Đàm Đạt)
- `XGBClassifier` với initial params
- 5-Fold CV → in accuracy mean ± std
- Hyperparameter tuning (Optuna hoặc manual):
  - `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`
- In best params + best accuracy
- **Markdown**: Params nào ảnh hưởng nhiều nhất? So sánh với baseline

#### 5.3 LightGBM (Tâm)
- `LGBMClassifier` với initial params
- 5-Fold CV → in accuracy mean ± std
- Hyperparameter tuning:
  - `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`
- In best params + best accuracy
- **Markdown**: So sánh với XGBoost về accuracy và tốc độ train

#### 5.4 CatBoost (Đàm Đạt)
- `CatBoostClassifier` — truyền danh sách `cat_features` nếu còn
- 5-Fold CV → in accuracy mean ± std
- Hyperparameter tuning:
  - `depth`, `learning_rate`, `iterations`
- In best params + best accuracy
- **Markdown**: CatBoost xử lý categorical tốt hơn không? So sánh 3 GBDT models

#### 5.5 Ensemble (Thành)
- **Weighted Average**: blend predictions từ XGBoost + LightGBM + CatBoost
  - Thử nhiều bộ weights → chọn tối ưu (dùng Optuna hoặc brute force trên CV)
- **Stacking** (optional): dùng LogisticRegression làm meta-learner trên OOF predictions
- 5-Fold CV cho ensemble → in accuracy
- **Markdown**: Ensemble có cải thiện so với single model tốt nhất? Bao nhiêu %?

#### 5.6 Bảng So Sánh Advanced
- Bảng tổng hợp gồm cả baseline lẫn advanced:

| Model | Accuracy | F1 | Ghi chú |
|---|---|---|---|
| Best Baseline (RF) | ... | ... | từ NB04 |
| XGBoost (tuned) | ... | ... | |
| LightGBM (tuned) | ... | ... | |
| CatBoost (tuned) | ... | ... | |
| **Ensemble** | ... | ... | **Best** |

- **Markdown**: Kết luận: model/ensemble nào sẽ dùng cho final submission? Tại sao?

---

## 🔶 GĐ4 — `06_evaluation.ipynb` (Đàm Đạt + Tuấn)

### Mục tiêu
Đánh giá final model + generate submission file + phân tích interpretability.

### Cấu trúc notebook

#### 6.1 Load Data & Train Final Model
- Load train final data
- Train best model/ensemble trên **toàn bộ train data** (không split)
- In confirmation: model đã train xong

#### 6.2 Detailed Evaluation (trên CV)
- **Confusion Matrix**: heatmap với annotated counts
- **Classification Report**: precision, recall, f1 cho mỗi class
- **ROC-AUC Curve**: vẽ ROC curve + tính AUC score
- **Markdown cho mỗi phần**: phân tích chi tiết — model sai ở đâu nhiều nhất? FP hay FN cao hơn? AUC có tốt không?

#### 6.3 Feature Importance / Interpretability
- **Feature Importance**: bar chart top 15 features (từ best model)
- **SHAP Analysis** (nếu có thời gian): SHAP summary plot, hoặc ít nhất waterfall plot cho vài samples
- **Markdown**: Top features nào quan trọng nhất? Có khớp với intuition từ EDA không? Giải thích ý nghĩa.

#### 6.4 So Sánh Tổng Hợp Tất Cả Models
- Bảng final gồm tất cả models từ NB04 + NB05
- Bar chart so sánh accuracy
- **Markdown**: Kết luận tổng thể về performance

#### 6.5 Generate Submission
- Load `test_final.csv`
- Predict bằng best model/ensemble
- Tạo DataFrame: `id`, `Depression`
- Lưu `outputs/submissions/submission.csv`
- In `.head()`, `.shape`, `value_counts()` của predictions
- **Markdown**: Kiểm tra: đúng 93,800 rows? Tỷ lệ Depression predicted có hợp lý so với train?

---

## 🔶 GĐ5 — Report & Video

### Report (LaTeX)
Cấu trúc bắt buộc 10 sections:

1. **Project Planning & Task Distribution** (Kiệt)
   - Bảng phân công rõ ràng: ai làm gì, deadline, trạng thái
   - Tỷ lệ % đóng góp mỗi thành viên

2. **Project Overview** (Tâm)
   - Giới thiệu cuộc thi, bài toán, dataset
   - Pipeline tổng thể (có thể vẽ diagram)

3. **Data Exploration** (Thành + Tuấn)
   - Tóm tắt key findings từ EDA
   - Chèn 4-5 biểu đồ quan trọng nhất kèm phân tích

4. **Data Preprocessing** (Kiệt)
   - Mô tả từng bước xử lý: missing values, encoding, outliers
   - Bảng before/after

5. **Feature Engineering** (Đàm Đạt + Tâm)
   - Mô tả features mới + lý do tạo
   - Bảng danh sách features cuối cùng

6. **Model Development** (Kiệt: Baseline, Thành: Advanced)
   - Mô tả từng model: params chính, cách train
   - Bảng tóm tắt models

7. **Experiments & Results** (Kiệt)
   - Bảng so sánh tất cả models (accuracy, F1, ...)
   - Confusion matrix, ROC curve
   - Phân tích: model nào tốt nhất và tại sao

8. **Conclusion & Future Work** (Tâm)
   - Tóm tắt những gì đã làm
   - Đề xuất cải thiện: more FE, neural networks, more data, ...

9. **Kaggle Screenshot** (Thành)
   - Screenshot kết quả submission

10. **References** (Kiệt)
    - Kaggle competition link
    - Thư viện sử dụng
    - Papers / blog posts tham khảo (nếu có)

### Video (ALL)
- 12-15 phút, 5 phần, mỗi người 1 phần
- Có thể dùng Google Meet recording
- Tải lên YouTube (public/unlisted)
