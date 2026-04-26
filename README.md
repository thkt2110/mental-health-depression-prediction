# Mental Health Depression Prediction

> **CSC17001 — Intelligent Data Analytics** | University of Science, VNU-HCM  
> Dự án dự đoán trầm cảm dựa trên khảo sát sức khỏe tâm thần sử dụng Machine Learning.  
> Nguồn dữ liệu: [Kaggle — Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11)

---

## Giới Thiệu

Dự án xây dựng mô hình phân loại nhị phân dự đoán liệu một cá nhân có bị **trầm cảm** (Depression = 1) hay không, dựa trên các yếu tố khảo sát như áp lực học tập/công việc, tình trạng giấc ngủ, căng thẳng tài chính, tiền sử gia đình và các chỉ số tâm lý khác.

**Kết quả tốt nhất**: XGBoost (tuned) đạt **Accuracy = 93.95%**, **F1 = 83.12%** trên 5-Fold Cross-Validation với 140,700 mẫu dữ liệu.

---

## Thành Viên Nhóm

| Thành viên | Vai trò chính |
|-----------|--------------|
| Thành | NB01 EDA, NB05 Advanced Modeling |
| Tuấn | NB01 EDA, NB04 Baseline, NB06 Feature Importance |
| Kiệt | NB02 Preprocessing, NB04 Baseline |
| Đàm Đạt | NB03 Feature Engineering, NB05 XGBoost, NB06 SHAP |
| Tâm | NB03 Feature Engineering, NB05 LightGBM, Demo predict() |

---

## Kết Quả Mô Hình

| Model | Accuracy (CV) | F1-Score | Ghi chú |
|-------|:------------:|:--------:|---------|
| Decision Tree (d=10) | 0.9292 ± 0.0010 | 0.8011 | Baseline |
| Random Forest (200) | 0.9364 ± 0.0014 | 0.8208 | Baseline |
| SVM (linear) | 0.9367 ± 0.0021 | 0.8230 | Baseline |
| Logistic Regression | 0.9376 ± 0.0019 | 0.8243 | Best Baseline |
| LightGBM (tuned) | 0.9392 ± 0.0013 | 0.8306 | Advanced |
| **XGBoost (tuned)** | **0.9395 ± 0.0016** | **0.8312** | **Best Overall** |

---

## Cấu Trúc Project

```
mental-health-depression-prediction/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Data Cleaning & Encoding
│   ├── 03_feature_engineering.ipynb  # Feature Creation & Selection
│   ├── 04_model_baseline.ipynb       # LR, Decision Tree, RF, SVM
│   ├── 05_model_advanced.ipynb       # XGBoost, LightGBM
│   └── 06_evaluation.ipynb           # SHAP, ROC-AUC, Demo predict()
├── src/
│   ├── eda_utils.py                  # Helper functions cho EDA
│   ├── preprocessing_utils.py        # Helper functions cho preprocessing
│   ├── modeling_utils.py             # Helper functions cho modeling
│   └── evaluation_utils.py           # Helper functions cho evaluation
├── data/
│   ├── raw/                          # Dữ liệu gốc (train.csv, test.csv)
│   └── processed/                    # Dữ liệu sau xử lý
├── report_latex/                     # Báo cáo LaTeX + PDF
├── submissions/
│   └── submission.csv                # File kết quả dự đoán
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Hướng Dẫn Cài Đặt

### 1. Tạo môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Tải dữ liệu từ [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e11/data) và đặt vào:
```
data/raw/train.csv
data/raw/test.csv
```

---

## Cách Chạy

Chạy các notebook **theo thứ tự** từ `01` đến `06`:

```
01_eda.ipynb               → Khám phá dữ liệu, phát hiện patterns
02_preprocessing.ipynb     → Làm sạch, encoding, merge cột conditional
03_feature_engineering.ipynb → Target encoding, interaction features, binning
04_model_baseline.ipynb    → Baseline models với 5-Fold CV
05_model_advanced.ipynb    → XGBoost & LightGBM tuning
06_evaluation.ipynb        → Đánh giá cuối, SHAP analysis, demo predict()
```

> **Lưu ý**: Mỗi notebook sử dụng `Kernel → Restart & Run All` trước khi nộp bài.

---

## Pipeline Tổng Quan

```
Raw Data (train.csv / test.csv)
        │
        ▼
[NB01] EDA
  - Phân tích 20 features, 140,700 mẫu
  - Phát hiện conditional missing (Student vs Working Professional)
  - Imbalanced target: 81.83% không trầm cảm
        │
        ▼
[NB02] Preprocessing
  - Drop id/Name, làm sạch giá trị bẩn
  - Merge cột conditional (Pressure, Satisfaction)
  - Binary encoding, Ordinal encoding
        │
        ▼
[NB03] Feature Engineering
  - Target Encoding cho City, Profession, Degree
  - Interaction features: pressure×stress, age×hours, risk_score
  - Age binning, Feature selection
        │
        ▼
[NB04] Baseline Models          [NB05] Advanced Models
  - Logistic Regression           - XGBoost (tuned)
  - Decision Tree                 - LightGBM (tuned)
  - Random Forest                 - So sánh với baseline
  - SVM
        │                               │
        └───────────────┬───────────────┘
                        ▼
                [NB06] Evaluation
                  - Confusion Matrix, ROC-AUC, Classification Report
                  - SHAP interpretability (summary, bar, waterfall)
                  - Demo hàm predict() với mock profiles
                  - Kaggle submission
```

---

## Thư Viện Sử Dụng

| Thư viện | Mục đích |
|---------|---------|
| `pandas`, `numpy` | Xử lý dữ liệu |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Baseline models, metrics, preprocessing |
| `xgboost`, `lightgbm` | Advanced models |
| `shap` | Model interpretability |
| `category_encoders` | Target Encoding |
| `pyarrow`, `fastparquet` | Lưu/đọc file Parquet |
| `jupyter`, `ipykernel` | Jupyter Notebook |
