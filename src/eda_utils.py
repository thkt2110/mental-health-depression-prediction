"""
Utility functions for Exploratory Data Analysis (EDA).
Used in Phase 1: 01_eda.ipynb
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _sanitize_filename(path):
    """Sanitize the filename part of a path to remove invalid characters."""
    dirpart = os.path.dirname(path)
    filepart = os.path.basename(path)
    # Replace characters invalid in Windows filenames with underscores
    filepart = re.sub(r'[\\/:*?"<>|]', '_', filepart)
    # Replace multiple spaces or underscores with single underscore
    filepart = re.sub(r'[\s_]+', '_', filepart).strip('_')
    return os.path.join(dirpart, filepart)


def _maybe_save(save_path):
    """Save current figure to save_path if provided."""
    if save_path is not None:
        save_path = _sanitize_filename(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


# ==========================================
# ZONE: THÀNH
# ==========================================
# Ghi chú: Các hàm tổng quan, univariate, bivariate, correlation heatmap...


def create_overview_table(df):
    """
    Tạo bảng tổng quan tất cả cột trong DataFrame.

    Returns:
        DataFrame gồm: Tên cột, Kiểu dữ liệu, Số missing, % missing, Số unique.
    """
    overview = pd.DataFrame({
        'Kiểu dữ liệu': df.dtypes,
        'Số missing': df.isnull().sum(),
        '% missing': (df.isnull().sum() / len(df) * 100).round(2),
        'Số unique': df.nunique(),
        'Giá trị mẫu': [df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else 'N/A' for col in df.columns]
    })
    return overview


def plot_target_distribution(df, target='Depression', save_path=None):
    """
    Vẽ countplot + pie chart cho biến mục tiêu.

    Args:
        df: DataFrame chứa dữ liệu.
        target: Tên cột target (mặc định 'Depression').
        save_path: Đường dẫn lưu hình (nếu có).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Countplot
    palette_target = {0: '#5DADE2', 1: '#E74C3C', '0': '#5DADE2', '1': '#E74C3C'}
    ax0 = sns.countplot(data=df, x=target, ax=axes[0], palette=palette_target, hue=target, legend=False)
    axes[0].set_title('Phân phối biến mục tiêu (Depression)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Depression (0: Không, 1: Có)', fontsize=12)
    axes[0].set_ylabel('Số lượng', fontsize=12)

    for p in ax0.patches:
        ax0.annotate(f'{int(p.get_height()):,}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Pie chart
    counts = df[target].value_counts().sort_index()
    labels = ['Không trầm cảm (0)', 'Trầm cảm (1)']
    colors = ['#5DADE2', '#E74C3C']
    explode = (0.03, 0.03)
    axes[1].pie(counts, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=explode,
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Tỷ lệ Depression', fontsize=14, fontweight='bold')

    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # In tỷ lệ
    print(f"Tỷ lệ Depression:")
    ratio = df[target].value_counts(normalize=True).sort_index()
    for idx, val in ratio.items():
        print(f"  {target} = {idx}: {val*100:.2f}%  ({df[target].value_counts().sort_index()[idx]:,} mẫu)")
    print(f"  Tỷ số (class 1 / class 0): {ratio[1]/ratio[0]:.3f}")


def plot_missing_values(df, save_path=None):
    """
    Vẽ bar chart % missing values cho các cột có missing.
    Kèm bảng thống kê.

    Args:
        df: DataFrame chứa dữ liệu.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Số missing': missing,
        '% missing': missing_pct
    })
    missing_df = missing_df[missing_df['Số missing'] > 0].sort_values('% missing', ascending=True)

    if missing_df.empty:
        print("✅ Không có missing values trong dataset!")
        return

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, max(5, len(missing_df) * 0.5)))
    bars = ax.barh(missing_df.index, missing_df['% missing'], color='#E74C3C', alpha=0.8)
    ax.set_title('Tỷ lệ Missing Values theo cột', fontsize=14, fontweight='bold')
    ax.set_xlabel('% Missing', fontsize=12)

    for bar, pct in zip(bars, missing_df['% missing']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center', fontsize=11)

    ax.set_xlim(0, missing_df['% missing'].max() * 1.15)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # Bảng thống kê
    print("\n📋 Chi tiết Missing Values:")
    display(missing_df.sort_values('% missing', ascending=False))


def plot_missing_pattern_by_group(df, group_col='Working Professional or Student',
                                  cols=None, save_path=None):
    """
    So sánh missing values giữa các nhóm (Student vs Working Professional).

    Args:
        df: DataFrame chứa dữ liệu.
        group_col: Cột phân nhóm.
        cols: Danh sách cột cần kiểm tra. Mặc định là các cột conditional.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    if cols is None:
        cols = ['Academic Pressure', 'CGPA', 'Study Satisfaction',
                'Work Pressure', 'Job Satisfaction']

    groups = df[group_col].unique()
    results = {}

    for group in groups:
        subset = df[df[group_col] == group]
        missing_count = subset[cols].isnull().sum()
        missing_pct = (missing_count / len(subset) * 100).round(2)
        results[group] = missing_pct

    result_df = pd.DataFrame(results)
    result_df.columns.name = group_col

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.6)))
    sns.heatmap(result_df, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': '% Missing'})
    ax.set_title(f'% Missing theo nhóm ({group_col})', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # Bảng chi tiết
    print(f"\n📋 Chi tiết missing theo nhóm:")
    for group in groups:
        subset = df[df[group_col] == group]
        print(f"\n--- {group} ({len(subset):,} mẫu) ---")
        for col in cols:
            n_miss = subset[col].isnull().sum()
            pct = n_miss / len(subset) * 100
            print(f"  {col}: {n_miss:,} missing ({pct:.1f}%)")


def plot_numerical_distribution(df, col, save_path=None):
    """
    Vẽ histogram + KDE + boxplot cho 1 numerical feature.
    In kèm thống kê describe + skewness.

    Args:
        df: DataFrame chứa dữ liệu.
        col: Tên cột numerical.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    data = df[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Histogram + KDE
    sns.histplot(data, kde=True, ax=axes[0], color='#3498DB', edgecolor='white', alpha=0.7)
    axes[0].axvline(data.mean(), color='#E74C3C', linestyle='--', linewidth=1.5, label=f'Mean = {data.mean():.2f}')
    axes[0].axvline(data.median(), color='#2ECC71', linestyle='--', linewidth=1.5, label=f'Median = {data.median():.2f}')
    axes[0].legend(fontsize=10)
    axes[0].set_title(f'Phân phối {col}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(col, fontsize=11)
    axes[0].set_ylabel('Tần suất', fontsize=11)

    # Boxplot
    sns.boxplot(data=df, y=col, ax=axes[1], color='#F39C12', width=0.4)
    axes[1].set_title(f'Boxplot {col}', fontsize=13, fontweight='bold')

    plt.suptitle(f'📊 Phân tích: {col}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # Thống kê
    skew = data.skew()
    kurt = data.kurtosis()
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outlier_low = data[data < (q1 - 1.5 * iqr)].shape[0]
    outlier_high = data[data > (q3 + 1.5 * iqr)].shape[0]
    print(f"  Skewness: {skew:.3f} | Kurtosis: {kurt:.3f}")
    print(f"  Mean: {data.mean():.2f} | Std: {data.std():.2f} | Min: {data.min():.2f} | Max: {data.max():.2f}")
    print(f"  Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
    print(f"  Outliers (IQR method): {outlier_low} thấp, {outlier_high} cao (tổng {outlier_low+outlier_high})")
    print("-" * 70)


def plot_categorical_distribution(df, col, top_n=15, save_path=None):
    """
    Vẽ countplot + frequency table cho 1 categorical feature.

    Args:
        df: DataFrame chứa dữ liệu.
        col: Tên cột categorical.
        top_n: Số lượng categories tối đa hiển thị (mặc định 15).
        save_path: Đường dẫn lưu hình (nếu có).
    """
    value_counts = df[col].value_counts()
    n_unique = len(value_counts)

    fig, ax = plt.subplots(figsize=(12, max(4, min(n_unique, top_n) * 0.4)))

    if n_unique > top_n:
        plot_data = value_counts.head(top_n)
        title = f'Top {top_n} — {col}  (tổng {n_unique} categories)'
    else:
        plot_data = value_counts
        title = f'Phân phối {col}  ({n_unique} categories)'

    colors = sns.color_palette('viridis', len(plot_data))
    bars = ax.barh(plot_data.index[::-1], plot_data.values[::-1], color=colors)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Số lượng', fontsize=11)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + len(df) * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{int(width):,}', va='center', fontsize=10)

    ax.set_xlim(0, plot_data.max() * 1.12)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # Frequency table
    freq = value_counts
    freq_pct = (freq / len(df) * 100).round(2)
    freq_table = pd.DataFrame({'Số lượng': freq, '% tổng': freq_pct})
    print(f"\n📋 Frequency Table — {col} (hiển thị top {min(20, n_unique)}):")
    display(freq_table.head(20))
    if n_unique > 20:
        print(f"  ... và {n_unique - 20} categories khác")
    print("=" * 70)


def plot_numerical_vs_target(df, col, target='Depression', save_path=None):
    """
    Vẽ boxplot + KDE overlay so sánh phân phối numerical feature
    giữa 2 nhóm Depression.

    Args:
        df: DataFrame chứa dữ liệu.
        col: Tên cột numerical.
        target: Tên cột target (mặc định 'Depression').
        save_path: Đường dẫn lưu hình (nếu có).
    """
    data = df[[col, target]].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    palette_target = {0: '#5DADE2', 1: '#E74C3C', '0': '#5DADE2', '1': '#E74C3C'}

    # Boxplot grouped by target
    sns.boxplot(data=data, x=target, y=col, ax=axes[0],
                palette=palette_target, hue=target, legend=False, width=0.5)
    axes[0].set_title(f'{col} theo Depression', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Depression (0: Không, 1: Có)', fontsize=11)
    axes[0].set_ylabel(col, fontsize=11)

    # KDE overlay
    for val, color, label in [(0, '#5DADE2', 'Không trầm cảm'),
                               (1, '#E74C3C', 'Trầm cảm')]:
        subset = data[data[target] == val][col]
        if len(subset) > 0:
            sns.kdeplot(subset, ax=axes[1], color=color, label=label,
                        linewidth=2, fill=True, alpha=0.3)
    axes[1].set_title(f'KDE — {col} phân biệt theo Depression', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(col, fontsize=11)
    axes[1].legend(fontsize=10)

    plt.suptitle(f'🔍 Bivariate: {col} vs Depression', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # Thống kê so sánh
    for val in [0, 1]:
        subset = data[data[target] == val][col]
        label = 'Không trầm cảm' if val == 0 else 'Trầm cảm'
        print(f"  {label}: mean={subset.mean():.2f}, median={subset.median():.2f}, std={subset.std():.2f}")
    print("-" * 70)


def plot_categorical_vs_target(df, col, target='Depression', top_n=15, save_path=None):
    """
    Vẽ grouped bar chart + tỷ lệ Depression theo từng category.

    Args:
        df: DataFrame chứa dữ liệu.
        col: Tên cột categorical.
        target: Tên cột target (mặc định 'Depression').
        top_n: Số categories tối đa hiển thị.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    data = df[[col, target]].dropna()

    # Lấy top categories theo tần suất
    top_cats = data[col].value_counts().head(top_n).index
    data_top = data[data[col].isin(top_cats)]

    # Tính tỷ lệ Depression cho mỗi category
    ct = pd.crosstab(data_top[col], data_top[target], normalize='index')
    ct = ct.reindex(top_cats)  # Giữ thứ tự theo tần suất

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(top_cats) * 0.35)))

    # Stacked bar chart (tỷ lệ %)
    colors = ['#5DADE2', '#E74C3C']
    ct.plot(kind='barh', stacked=True, ax=axes[0], color=colors, edgecolor='white')
    axes[0].set_title(f'Tỷ lệ Depression theo {col}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tỷ lệ', fontsize=11)
    axes[0].set_ylabel('')
    axes[0].legend(['Không trầm cảm (0)', 'Trầm cảm (1)'], fontsize=10, loc='lower right')

    # Bar chart chỉ tỷ lệ Depression = 1
    if 1 in ct.columns:
        depression_rate = ct[1].sort_values(ascending=True)
        bars = axes[1].barh(depression_rate.index, depression_rate.values,
                            color='#E74C3C', alpha=0.8)
        axes[1].set_title(f'Tỷ lệ trầm cảm theo {col}', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Tỷ lệ Depression = 1', fontsize=11)
        axes[1].set_ylabel('')

        for bar in bars:
            width = bar.get_width()
            axes[1].text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                         f'{width:.1%}', va='center', fontsize=10)
        axes[1].set_xlim(0, depression_rate.max() * 1.15)

    n_unique = data[col].nunique()
    suffix = f'  (hiển thị top {top_n}/{n_unique})' if n_unique > top_n else ''
    plt.suptitle(f'🔍 Bivariate: {col} vs Depression{suffix}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()
    print("-" * 70)


def plot_correlation_heatmap(corr_matrix, figsize=(14, 10), save_path=None):
    """
    Vẽ correlation heatmap.

    Args:
        corr_matrix: Ma trận tương quan (output từ df.corr()).
        figsize: Kích thước biểu đồ.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5, square=True,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
    ax.set_title('Ma trận tương quan Pearson — Numerical Features',
                 fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()


def plot_top_correlations(corr_matrix, target='Depression', top_n=10, save_path=None):
    """
    Vẽ bar chart top features tương quan mạnh nhất với target.

    Args:
        corr_matrix: Ma trận tương quan.
        target: Tên biến target.
        top_n: Số lượng features hiển thị.
        save_path: Đường dẫn lưu hình (nếu có).
    """
    if target not in corr_matrix.columns:
        print(f"⚠️ Cột '{target}' không có trong ma trận tương quan!")
        return

    target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in target_corr.values]

    bars = ax.barh(target_corr.index[::-1], target_corr.values[::-1],
                   color=colors[::-1], alpha=0.85, edgecolor='white')
    ax.set_title(f'Top {top_n} Features — Tương quan với {target}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (0.005 if width >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', va='center',
                ha='left' if width >= 0 else 'right', fontsize=10)

    plt.tight_layout()
    _maybe_save(save_path)
    plt.show()

    # In bảng
    print(f"\n📋 Tương quan với {target}:")
    for feat, val in target_corr.items():
        direction = '↑ Dương' if val > 0 else '↓ Âm'
        print(f"  {feat:30s}: {val:+.4f}  ({direction})")


# ==========================================
# ZONE: TUẤN
# ==========================================
# Ghi chú: Các hàm Data quality check, Univariate analysis...




# ==========================================
# ZONE: KIỆT
# ==========================================
# Ghi chú: Functions từ list insight của Kaggle (nếu có)




# ==========================================
# ZONE: ĐÀM ĐẠT
# ==========================================
# Ghi chú: Functions từ list insight của Kaggle (nếu có)




# ==========================================
# ZONE: TÂM
# ==========================================
# Ghi chú: Các hàm phục vụ Kiểm định thống kê (Chi-Square, T-Test...)



