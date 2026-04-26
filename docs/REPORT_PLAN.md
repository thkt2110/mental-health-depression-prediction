# 📄 Yêu Cầu Nội Dung Báo Cáo

> Dựa theo `require_lab.txt` — CSC17001 Intelligent Data Analytics  
> Báo cáo cần: **giải thích chi tiết từng bước + hình ảnh minh họa + sơ đồ + phương trình**  
> Format: LaTeX → PDF

---

## Section 1 — Project Planning & Task Distribution

**Mục đích**: Minh bạch hóa đóng góp từng thành viên để tính điểm cá nhân.

**Cần có:**
- Bảng phân công: mỗi thành viên phụ trách task nào / notebook nào
- Tỷ lệ % hoàn thành của từng người (do nhóm tự đánh giá)
- Công thức tính điểm cá nhân: `điểm cá nhân = điểm nhóm × % đóng góp`  
  Ví dụ: nhóm được 9.0, A đóng góp 90% → A nhận 9.0 × 90% = 8.1
- Không cần dài, chỉ cần rõ ràng và trung thực

---

## Section 2 — Project Overview

**Mục đích**: Giới thiệu ngắn gọn về dự án để người đọc nắm được bức tranh tổng thể.

**Cần có:**
- Nguồn gốc bài toán: lấy cảm hứng từ Kaggle competition "Exploring Mental Health Data"
- Phát biểu bài toán: dự đoán một người có bị trầm cảm (Depression = 0/1) dựa trên dữ liệu khảo sát
- Loại bài toán: Binary Classification
- Metric đánh giá: Accuracy
- Mô tả dataset ngắn: nguồn gốc, kích thước (train/test), tính chất (synthetic)
- **Sơ đồ pipeline tổng thể** (hình vẽ hoặc diagram):  
  `Raw Data → EDA → Preprocessing → Feature Engineering → Modeling → Evaluation`

---

## Section 3 — Data Exploration

**Mục đích**: Mô tả sự hiểu biết về dữ liệu, các quan sát chính và insights.

**Cần có:**

### 3.1 Tổng quan Dataset
- Số lượng features, số dòng train/test
- Bảng danh sách features: tên cột — kiểu dữ liệu (numerical/categorical) — ý nghĩa
- Tỷ lệ missing values theo từng cột (bảng hoặc hình)

### 3.2 Phân tích Target Variable
- **Hình**: countplot hoặc pie chart phân phối Depression = 0 vs 1
- Nhận xét: dataset có bị imbalanced không?

### 3.3 Phân tích Features
- **Numerical features**: phân phối (histogram/KDE), thống kê mô tả (mean, std, min, max)
- **Categorical features**: tần suất xuất hiện (countplot), số lượng unique values
- **Hình**: chọn 4–6 biểu đồ đại diện nhất (không cần vẽ hết, chọn những cái nổi bật nhất)

### 3.4 Bivariate Analysis — Feature vs Target
- **Hình**: mối quan hệ từng feature với Depression (boxplot / stacked bar / violin plot)
- Nhận xét: feature nào có khả năng phân biệt tốt Depression?

### 3.5 Correlation Analysis
- **Hình**: Correlation heatmap của numerical features
- Nhận xét: features nào tương quan mạnh với target? Có multicollinearity không?

### 3.6 Multivariate Analysis
- Phân tích tương tác đa biến giữa các features quan trọng:
  - `Sleep Duration × Financial Stress → Depression`: hiệu ứng cộng gộp làm tăng tỷ lệ trầm cảm
  - `Suicidal Thoughts × Family History → Depression`: tỷ lệ trầm cảm >50% khi cả hai đều có
  - `Work/Study Hours × Age Group → Depression`: pattern khác nhau theo nhóm tuổi
- **Hình**: heatmap / crosstab minh họa 2–3 interaction patterns quan trọng nhất
- Nhận xét: các patterns này là **tiền đề trực tiếp** cho việc tạo interaction features ở Section 5

### 3.7 Key Findings
- Danh sách 5–7 findings quan trọng rút ra từ EDA
- Đề xuất hướng xử lý cho Preprocessing và Feature Engineering

---

## Section 4 — Data Preprocessing

**Mục đích**: Làm sạch dữ liệu, chuẩn bị cho modeling.

**Cần có:**

### 4.1 Làm Sạch Giá Trị Bẩn (Data Cleaning — Noisy Data)
> **Yêu cầu đề bài**: *"Data cleaning, transformation, handling of missing or **noisy** data"*

- Phát hiện dữ liệu bẩn (noisy data) trong 2 cột:
  - `Sleep Duration`: 36 unique values thay vì 4 — ~79 mẫu chứa tên thành phố, tên cột (`Pune`, `Work_Study_Hours`) bị swap vào
  - `Dietary Habits`: 23 unique values thay vì 3 — ~27 mẫu chứa tên bằng cấp, giới tính (`BSc`, `Male`) bị swap vào
- Nguyên nhân: data entry error / column swap trong quá trình thu thập
- Chiến lược xử lý: map các giá trị bẩn về category hợp lệ gần nhất, phần còn lại → mode
- **Bảng**: kết quả trước/sau cleaning (số unique values, ví dụ giá trị bẩn)

### 4.2 Xử lý Missing Values
- Bảng: cột nào bị missing, tỷ lệ bao nhiêu, chiến lược xử lý là gì
- Phân loại missing mechanism: **MNAR (Missing Not At Random)** cho 5 cột conditional
  - `Academic Pressure`, `CGPA`, `Study Satisfaction` chỉ có ở Student
  - `Work Pressure`, `Job Satisfaction` chỉ có ở Working Professional

### 4.3 Merge Conditional Columns
> Đây là quyết định thiết kế phức tạp nhất, thể hiện sự hiểu biết sâu về domain (sinh viên vs người đi làm).

- Chiến lược merge:
  - `Academic Pressure` + `Work Pressure` → `Pressure`
  - `Study Satisfaction` + `Job Satisfaction` → `Satisfaction`
  - `CGPA` giữ nguyên + thêm binary flag `has_cgpa`
- Kết quả: từ 20 cột → 17 cột
- Giải thích lý do: không thể fill bằng mean/median vì đó là MNAR — giá trị missing **có ý nghĩa** (sinh viên không có Work Pressure, người đi làm không có Academic Pressure)

### 4.4 Loại bỏ Features không cần thiết
- Drop `Name` (không có nghĩa dự đoán) — giải thích lý do

### 4.5 Encoding Categorical Variables
- Binary encoding: `Gender`, `Suicidal Thoughts`, `Family History`, `Working Professional or Student`
- Ordinal encoding: `Sleep Duration` (thứ tự giờ ngủ), `Dietary Habits` (Unhealthy < Moderate < Healthy)
- Giải thích lý do chọn từng loại encoding

### 4.6 Feature Scaling
- Giải thích khi nào cần scaling: chỉ cần cho Logistic Regression và SVM, không cần cho tree-based models
- **Phương trình bắt buộc** (LaTeX):

$$z = \frac{x - \mu}{\sigma} \qquad \text{(Standardization)}$$

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \qquad \text{(Min-Max Normalization)}$$

### 4.7 Kết quả Preprocessing
- Bảng: số features / số dòng trước và sau xử lý
- Xác nhận: không còn missing values, không còn noisy data

---

## Section 5 — Feature Engineering

**Mục đích**: Trình bày cách tạo hoặc chọn features để cải thiện hiệu suất model.

**Cần có:**

### 5.1 Encoding High Cardinality Features
- `City`, `Profession`, `Degree` có quá nhiều categories → dùng Target Encoding
- **Phương trình bắt buộc** (LaTeX):

$$\text{TargetEnc}(c) = \frac{\displaystyle\sum_{i:\, x_i = c} y_i + \lambda \cdot \bar{y}}{n_c + \lambda}$$

Giải thích: $n_c$ = số lần category $c$ xuất hiện, $\bar{y}$ = mean target toàn bộ, $\lambda$ = smoothing  
Giải thích tại sao cần CV khi fit Target Encoding (tránh data leakage)

### 5.2 Phân Nhóm Tuổi (Age Binning)
- Kỹ thuật: Discretization / Binning — chia `Age` thành 4 nhóm ordinal
  - Nhóm 1: 18–29 tuổi
  - Nhóm 2: 30–42 tuổi
  - Nhóm 3: 43–51 tuổi
  - Nhóm 4: 52+ tuổi
- Tạo feature `age_group` dạng ordinal encoding
- Justification: dựa trên phân tích EDA (Section 3) cho thấy pattern trầm cảm khác nhau theo nhóm tuổi

### 5.3 Tạo Interaction Features
- Liệt kê các features tạo mới, **kèm công thức** và lý do (dựa trên Multivariate Analysis — Section 3.6):

$$\text{stress\_load} = \text{Pressure} \times \text{Financial\_Stress}$$

$$\text{wellbeing\_index} = \frac{\text{Satisfaction} + \text{Sleep\_Duration}}{2}$$

$$\text{stress\_satisfaction\_ratio} = \frac{\text{Pressure}}{\text{Satisfaction} + 1}$$

$$\text{Risk\_Score} = \text{Suicidal\_Thoughts} + \text{Family\_History}$$

- **Hình**: phân phối features mới + mối quan hệ với Depression

### 5.4 Feature Selection
- Phương pháp: feature importance từ Random Forest / Mutual Information / correlation
- **Hình**: bar chart feature importance hoặc correlation với target
- Kết luận: giữ lại bao nhiêu features, loại bỏ những gì và tại sao

---

## Section 6 — Model Development

**Mục đích**: Tóm tắt các models đã sử dụng, bao gồm parameters, kiến trúc và thư viện.

> **Yêu cầu của thầy**: "Summarize the models used, including **parameters, architectures, or libraries** applied. Multiple models can be used."

**Cần có:**

### 6.1 Cài đặt thực nghiệm chung
- Validation strategy: 5-Fold Stratified Cross Validation — giải thích lý do chọn Stratified
- Metrics: Accuracy (chính) + F1, Precision, Recall (phụ)
- Libraries: scikit-learn, xgboost, lightgbm
- **Phương trình Evaluation Metrics** (LaTeX):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 6.2 Baseline Models

Với **mỗi model**, trình bày:
- Mô tả ngắn cách hoạt động
- **Phương trình / công thức cốt lõi** (bắt buộc)
- Hyperparameters chính đã dùng

| Model | Phương trình cần chèn |
|---|---|
| **Logistic Regression** | $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x}+b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}$ và Binary Cross-Entropy loss |
| **Decision Tree** | $\text{Gini}(t) = 1 - \sum_k p_k^2$ và $\text{Entropy}(t) = -\sum_k p_k \log_2 p_k$ |
| **Random Forest** | $\hat{y} = \text{majority\_vote}(h_1(\mathbf{x}), \ldots, h_T(\mathbf{x}))$ |
| **SVM** | $\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 \;\text{ s.t. }\; y_i(\mathbf{w}^T\mathbf{x}_i+b) \geq 1$ |

### 6.3 Advanced Models

| Model | Phương trình cần chèn |
|---|---|
| **XGBoost** | $\mathcal{L}(\theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$ |
| **LightGBM** | $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$ |

- Giải thích điểm khác biệt XGBoost vs LightGBM: **Level-wise** (XGB) vs **Leaf-wise** growth (LGBM)
- **Hình (nếu có)**: minh họa sự khác nhau giữa 2 chiến lược growth

---

## Section 7 — Experiments & Results

**Mục đích**: Báo cáo kết quả training/testing, so sánh models và phân tích.

**Cần có:**

### 7.1 Bảng So Sánh Tất Cả Models
Điền số liệu thực tế từ notebook:

| Model | Accuracy (CV mean ± std) | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |
| SVM | | | | |
| XGBoost | | | | |
| LightGBM | | | | |

### 7.2 Confusion Matrix
- **Phương trình** (LaTeX):

$$\text{Confusion Matrix} = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

- **Hình**: Confusion matrix của best model
- Phân tích: True Positive, False Negative, v.v.
- Giải thích ý nghĩa thực tế: trong bài toán phát hiện trầm cảm, False Negative (bỏ sót người bệnh) nguy hiểm hơn False Positive

### 7.3 Classification Report
- **Bảng**: Per-class Precision / Recall / F1 cho cả class 0 (Not Depressed) và class 1 (Depressed)
- Phân tích trade-off Precision vs Recall:
  - Class 1 (Depressed) cần **Recall cao** → phát hiện hết ca bệnh
  - Class 0 (Not Depressed) cần **Precision cao** → giảm false alarm
- Nhận xét: model có xu hướng thiên về class nào? Có cần điều chỉnh threshold không?

### 7.4 ROC-AUC Curve
- **Phương trình** (LaTeX):

$$\text{AUC} = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)$$

- **Hình**: ROC curve
- Ghi AUC score — giải thích AUC = X có nghĩa gì

### 7.5 Feature Importance & SHAP Analysis

#### 7.5a Feature Importance (Built-in)
- **Hình**: Feature importance bar chart (top 15 features) — so sánh XGBoost vs LightGBM
- Nhận xét sự khác biệt giữa 2 model

#### 7.5b SHAP Summary Plot
- **Hình**: SHAP summary plot (dot plot)
- Phân tích: feature đó tác động theo chiều nào đến Depression?

#### 7.5c SHAP Bar Plot
- **Hình**: SHAP bar plot (mean |SHAP|)
- So sánh ranking features theo SHAP vs built-in importance

#### 7.5d SHAP Waterfall — Case Study
- **Hình**: Waterfall plot cho 2 ca cụ thể:
  - Ca A: người bị trầm cảm — giải thích từng feature đóng góp thế nào vào prediction
  - Ca B: người không bị trầm cảm — so sánh sự khác biệt
- Ý nghĩa: thể hiện model interpretability ở mức **individual prediction**, giải thích tại sao model đưa ra quyết định cụ thể cho từng ca
- Có khớp với intuition từ EDA không?

### 7.6 Test Predictions & Submission
- Dự đoán trên test set (93,800 mẫu) sử dụng best model
- Tạo file `submission.csv` theo format Kaggle (`id`, `Depression`)
- Chứng minh model hoạt động **end-to-end**: từ raw data → preprocessing → feature engineering → prediction cuối cùng

### 7.7 Nhận xét & Phân tích
- Model nào cho kết quả tốt nhất? Tại sao?
- Advanced models cải thiện bao nhiêu so với baseline?
- Điểm mạnh và hạn chế của approach hiện tại

---

## Section 8 — Conclusion & Future Work

**Mục đích**: Tóm tắt findings và đề xuất hướng cải thiện.

**Cần có:**

### 8.1 Conclusions
- Tóm tắt pipeline đã thực hiện (1 đoạn ngắn)
- Model tốt nhất là gì, đạt accuracy bao nhiêu
- 3–5 insights quan trọng nhất rút ra từ dự án (kết hợp EDA + SHAP)

### 8.2 Future Work
Đề xuất ít nhất 4–5 hướng cải thiện, mỗi hướng có giải thích ngắn:
- Xử lý imbalanced data tốt hơn (SMOTE, class weighting)
- Thử neural networks (MLP, TabNet) cho tabular data
- Thu thập thêm dữ liệu thực tế (không phải synthetic)
- Thêm features từ nguồn bên ngoài (kinh tế xã hội, khí hậu vùng)
- Triển khai model thành ứng dụng sàng lọc rủi ro trầm cảm

---

## Section 9 — References

**Mục đích**: Liệt kê tất cả nguồn tài liệu đã tham khảo.

**Yêu cầu**: "Include any references in a **properly formatted bibliography** section."

**Cần có** (tối thiểu 5 references, format APA hoặc IEEE nhất quán):

```
[1] Kaggle. (2024). Playground Series - Season 4, Episode 11: Exploring Mental Health Data.
    https://www.kaggle.com/competitions/playground-series-s4e11

[2] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining (pp. 785–794).

[3] Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient
    Gradient Boosting Decision Tree. In Advances in Neural Information Processing
    Systems (NeurIPS), 30.

[4] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model
    Predictions. In Advances in Neural Information Processing Systems (NeurIPS), 30.

[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
    Journal of Machine Learning Research, 12, 2825–2830.

[6] McKinney, W. (2010). Data Structures for Statistical Computing in Python.
    In Proceedings of the 9th Python in Science Conference (SciPy 2010).
```

---

## 🖼️ Danh Sách Hình Cần Có (tối thiểu)

| Section | Hình cần có | Ghi chú |
|---|---|---|
| 2 | Pipeline diagram | Tự vẽ (LaTeX tikz / draw.io / hình từ NB) |
| 3 | Target distribution | Từ NB01 |
| 3 | 2–3 bivariate plots | Từ NB01 (chọn đẹp nhất) |
| 3 | Correlation heatmap | `docs/feat_corr.png` ✅ có sẵn |
| 4 | Missing values visualization | Từ NB02 |
| 5 | Feature importance / correlation | `docs/feat_importance.png` ✅ có sẵn |
| 5 | Interaction features vs Depression | `docs/interaction_*.png` ✅ có sẵn |
| 6 | (Optional) Minh họa Leaf-wise vs Level-wise | Hình từ internet hoặc tự vẽ |
| 7 | Confusion Matrix | Từ NB06 |
| 7 | ROC-AUC Curve | Từ NB06 |
| 7 | Feature importance advanced | `docs/advanced_feature_importance.png` ✅ có sẵn |
| 7 | SHAP summary plot | Từ NB06 |

---

## ✅ Checklist Cuối Trước Khi Nộp

**Nội dung:**
- [ ] Đủ 9 sections (Planning → References)
- [ ] Mỗi section có ít nhất 1 hình hoặc bảng minh họa
- [ ] Section 3 có Multivariate Analysis (tiền đề cho Feature Engineering)
- [ ] Section 4 có Data Cleaning (noisy data) + Merge Conditional Columns tách riêng
- [ ] Tất cả công thức toán học đã có trong Section 4, 5, 6, 7 (scaling, interaction features, models, evaluation metrics, CM, AUC)
- [ ] Section 5 có Age Binning + công thức Interaction Features
- [ ] Bảng so sánh models (Section 7) có đủ số liệu thực tế từ notebook
- [ ] Section 7 có Classification Report (per-class Precision/Recall/F1)
- [ ] SHAP analysis có đủ 4 loại: Feature Importance, Summary, Bar, Waterfall (2 ca)
- [ ] Section 7 có Test Predictions & Submission
- [ ] Future Work có ít nhất 4 ý cụ thể
- [ ] References đúng format, ≥ 5 mục

**Format:**
- [ ] YouTube video link có trong báo cáo (Section 2 hoặc trang bìa)
- [ ] Compile PDF thành công, không lỗi
- [ ] File đặt tên đúng: `TeamName_Report.pdf`
- [ ] Kích thước ZIP < 20MB (hoặc có Google Drive link)
