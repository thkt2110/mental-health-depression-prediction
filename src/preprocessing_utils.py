

# ============================================================
# ZONE: THÀNH
# ============================================================




# ============================================================
# ZONE: TUẤN
# ============================================================




# ============================================================
# ZONE: KIỆT
# ============================================================
# Ghi chú: Các hàm xử lý Missing Value, Encoding, Scaling...
"""
Utility functions for Data Preprocessing & Feature Engineering.
Used in Phase 2: 02_preprocessing.ipynb & 03_feature_engineering.ipynb

Functions are organized by preprocessing stage:
  1. Column management
  2. Dirty value cleaning
  3. Missing value handling
  4. Binary encoding
  5. Conditional column merging
  6. Ordinal encoding
  7. Validation
  8. Persistence
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# ============================================================
# STAGE 1 — COLUMN MANAGEMENT
# ============================================================

def extract_ids(df: pd.DataFrame, id_col: str = "id") -> pd.Series:
    """
    Extract the identifier column from a DataFrame before dropping it.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing the identifier column.
    id_col : str
        Name of the identifier column (default: "id").

    Returns
    -------
    pd.Series
        The identifier series, with the same index as df.
    """
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame.")
    return df[id_col].copy()


def drop_irrelevant_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Drop columns that carry no predictive information.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols_to_drop : list of str
        Column names to remove.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed. Only drops columns
        that actually exist — silently skips missing column names so
        the function is safe to call on both train and test.
    """
    existing = [c for c in cols_to_drop if c in df.columns]
    dropped = df.drop(columns=existing)
    print(f"  Dropped {len(existing)} column(s): {existing}")
    return dropped


# ============================================================
# STAGE 2 — DIRTY VALUE CLEANING
# ============================================================

def clean_categorical_column(
    df: pd.DataFrame,
    col: str,
    valid_categories: set,
    fill_value=None,
) -> pd.DataFrame:
    """
    Replace any value in `col` that is not in `valid_categories` with NaN,
    then fill NaN with `fill_value` (or the column mode if fill_value is None).

    This handles both pre-existing NaN and dirty values (misplaced data from
    other columns, free-text entries, etc.) in a single pass.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the column to clean.
    valid_categories : set
        The set of acceptable string values for this column.
    fill_value : scalar or None
        Value used to replace NaN after invalidation.
        If None, the mode of the remaining valid values is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with `col` containing only valid categories.
    """
    df = df.copy()

    before_dirty = (~df[col].isin(valid_categories) & df[col].notna()).sum()
    before_nan   = df[col].isna().sum()

    # Invalidate values outside the valid set
    df[col] = df[col].where(df[col].isin(valid_categories), other=np.nan)

    # Determine fill value
    if fill_value is None:
        fill_value = df[col].mode(dropna=True).iloc[0]

    df[col] = df[col].fillna(fill_value)

    after_nan = df[col].isna().sum()
    print(
        f"  [{col}] dirty values replaced: {before_dirty} | "
        f"pre-existing NaN: {before_nan} | "
        f"filled with '{fill_value}' | remaining NaN: {after_nan}"
    )
    return df


# ============================================================
# STAGE 3 — MISSING VALUE HANDLING
# ============================================================

def fill_sparse_missing(
    df: pd.DataFrame,
    col: str,
    strategy: str = "median",
    constant=None,
) -> pd.DataFrame:
    """
    Fill sparse missing values (very few NaN) in a single column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column to fill.
    strategy : str
        One of 'median', 'mode', 'constant'.
    constant : scalar or None
        Used when strategy='constant'. Ignored otherwise.

    Returns
    -------
    pd.DataFrame
        DataFrame with the column's NaN filled.
    """
    df = df.copy()
    before = df[col].isna().sum()

    if before == 0:
        print(f"  [{col}] No missing values — skipping.")
        return df

    if strategy == "median":
        fill_value = df[col].median()
    elif strategy == "mode":
        fill_value = df[col].mode(dropna=True).iloc[0]
    elif strategy == "constant":
        if constant is None:
            raise ValueError("strategy='constant' requires a non-None `constant` argument.")
        fill_value = constant
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: median, mode, constant.")

    df[col] = df[col].fillna(fill_value)
    after = df[col].isna().sum()
    print(f"  [{col}] Filled {before} NaN with {strategy}='{fill_value}' | remaining NaN: {after}")
    return df


# ============================================================
# STAGE 4 — BINARY ENCODING
# ============================================================

def encode_binary_features(df: pd.DataFrame, binary_mappings: dict) -> pd.DataFrame:
    """
    Apply explicit 0/1 label encoding to binary categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    binary_mappings : dict
        Mapping of the form:
            {
                "column_name": {"category_a": 0, "category_b": 1},
                ...
            }
        Columns are replaced in-place with integer dtype.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns encoded as int64.

    Notes
    -----
    Any value not present in the mapping dict is mapped to NaN.
    This surfaces unexpected category values rather than silently
    encoding them to a wrong integer.
    """
    df = df.copy()
    for col, mapping in binary_mappings.items():
        if col not in df.columns:
            print(f"  [{col}] Not found — skipping.")
            continue
        unexpected = set(df[col].dropna().unique()) - set(mapping.keys())
        if unexpected:
            print(f"  [{col}] WARNING: unexpected values {unexpected} — will become NaN.")
        df[col] = df[col].map(mapping).astype("Int64")  # nullable int for safety
        print(f"  [{col}] Encoded: {mapping}")
    return df


# ============================================================
# STAGE 5 — CONDITIONAL COLUMN MERGING
# ============================================================

def merge_conditional_columns(
    df: pd.DataFrame,
    role_col: str = "Working Professional or Student",
) -> pd.DataFrame:
    """
    Merge the student-only and professional-only conditional columns into
    unified features that cover all rows without introducing fake values.

    Merge logic
    -----------
    - `Pressure`    = Academic Pressure  (student rows, role==0)
                    = Work Pressure      (professional rows, role==1)
    - `Satisfaction`= Study Satisfaction (student rows, role==0)
                    = Job Satisfaction   (professional rows, role==1)
    - `has_cgpa`    = 1 if student, 0 if professional  (flag)
    - `CGPA`        = filled with 0.0 for professional rows
                      (professionals genuinely have no CGPA)

    The four original conditional columns are dropped after verification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must already have `role_col` encoded as integer
        (0 = Student, 1 = Working Professional).
    role_col : str
        Name of the already-encoded professional/student column.

    Returns
    -------
    pd.DataFrame
        DataFrame with `Pressure`, `Satisfaction`, `has_cgpa` added and
        the four original conditional columns removed.

    Raises
    ------
    AssertionError
        If `Pressure` or `Satisfaction` contains NaN after the merge,
        which would indicate an unexpected data issue.
    """
    df = df.copy()

    is_professional = df[role_col] == 1   # boolean mask

    # --- Pressure ---
    df["Pressure"] = np.where(
        is_professional,
        df["Work Pressure"],
        df["Academic Pressure"],
    ).astype(float)

    # --- Satisfaction ---
    df["Satisfaction"] = np.where(
        is_professional,
        df["Job Satisfaction"],
        df["Study Satisfaction"],
    ).astype(float)

    # Fill any remaining NaN from unexpected gaps with median values
    p_nan = df["Pressure"].isna().sum()
    s_nan = df["Satisfaction"].isna().sum()
    if p_nan > 0:
        p_median = df["Pressure"].median(skipna=True)
        df["Pressure"] = df["Pressure"].fillna(p_median)
        print(f"  [WARN] Pressure had {p_nan} NaN; filled with median={p_median:.2f}.")
    if s_nan > 0:
        s_median = df["Satisfaction"].median(skipna=True)
        df["Satisfaction"] = df["Satisfaction"].fillna(s_median)
        print(f"  [WARN] Satisfaction had {s_nan} NaN; filled with median={s_median:.2f}.")

    # --- CGPA flag & fill ---
    # has_cgpa = 1 for students (they have real CGPA), 0 for professionals (sentinel)
    df["has_cgpa"] = (~is_professional).astype(int)

    # Fill NaN CGPA differently per group:
    #   Professionals → 0.0  (sentinel: "not applicable")
    #   Students      → student CGPA median (the rare student with missing CGPA
    #                   should not receive the professional sentinel of 0.0)
    is_student = ~is_professional
    cgpa_null = df["CGPA"].isna()
    student_cgpa_median = df.loc[is_student & ~cgpa_null, "CGPA"].median()

    student_nan_count = (cgpa_null & is_student).sum()
    prof_nan_count    = (cgpa_null & is_professional).sum()

    if student_nan_count > 0:
        df.loc[cgpa_null & is_student, "CGPA"] = student_cgpa_median
        print(f"  [WARN] {student_nan_count} student(s) had NaN CGPA; filled with student median={student_cgpa_median:.2f}.")

    df.loc[cgpa_null & is_professional, "CGPA"] = 0.0
    if prof_nan_count > 0:
        print(f"  Filled {prof_nan_count} professional CGPA NaN with sentinel 0.0.")

    # --- Verify before dropping originals ---
    p_nan = df["Pressure"].isna().sum()
    s_nan = df["Satisfaction"].isna().sum()
    assert p_nan == 0, f"Pressure still has {p_nan} NaN after merge."
    assert s_nan == 0, f"Satisfaction still has {s_nan} NaN after merge."

    # --- Drop originals ---
    cols_to_drop = [
        "Academic Pressure", "Work Pressure",
        "Study Satisfaction", "Job Satisfaction",
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)

    print(f"  Merged → 'Pressure' (NaN: {p_nan}), 'Satisfaction' (NaN: {s_nan})")
    print(f"  Created → 'has_cgpa' (students: {df['has_cgpa'].sum()}, professionals: {(df['has_cgpa']==0).sum()})")
    print(f"  Dropped original conditional columns: {existing}")
    return df


# ============================================================
# STAGE 6 — ORDINAL ENCODING
# ============================================================

def encode_ordinal_features(
    df: pd.DataFrame,
    ordinal_mappings: dict,
) -> pd.DataFrame:
    """
    Encode ordinal categorical columns using an explicit ordered mapping.

    The integer assigned to each category reflects its position in the
    provided ordered list (0 = lowest, len-1 = highest).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    ordinal_mappings : dict
        Mapping of the form:
            {
                "column_name": ["lowest_category", ..., "highest_category"],
            }
        The list order determines the encoded integer value.

    Returns
    -------
    pd.DataFrame
        DataFrame with ordinal columns encoded as int64.

    Notes
    -----
    Any value not present in the ordered list is mapped to NaN.
    After cleaning (Stage 2), this should never occur.
    """
    df = df.copy()
    for col, ordered_categories in ordinal_mappings.items():
        if col not in df.columns:
            print(f"  [{col}] Not found — skipping.")
            continue
        mapping = {cat: idx for idx, cat in enumerate(ordered_categories)}
        unexpected = set(df[col].dropna().unique()) - set(mapping.keys())
        if unexpected:
            print(f"  [{col}] WARNING: unexpected values after cleaning: {unexpected}")
        df[col] = df[col].map(mapping).astype("Int64")
        print(f"  [{col}] Ordinal encoding applied: {mapping}")
    return df


# ============================================================
# STAGE 7 — VALIDATION
# ============================================================

def validate_preprocessed_data(
    df: pd.DataFrame,
    expected_columns: list,
    high_cardinality_cols: list,
    is_train: bool = True,
    nullable_cols: list = None,
) -> None:
    """
    Run a battery of assertions to confirm the DataFrame meets the
    data contract required before saving.

    Checks performed
    ----------------
    1. All expected columns are present (no missing, no extra).
    2. Zero NaN in any column except those listed in `nullable_cols`.
    3. High-cardinality columns retain object dtype.
    4. Encoded columns contain only expected integer values.
    5. Merged columns (`Pressure`, `Satisfaction`) are in valid range.
    6. Dropped columns are absent.
    7. Depression column untouched (train only).

    Parameters
    ----------
    df : pd.DataFrame
        The processed DataFrame to validate.
    expected_columns : list of str
        The complete list of column names expected in the output.
    high_cardinality_cols : list of str
        Columns that are intentionally left as object dtype (strings).
    is_train : bool
        True if validating the training set (includes 'Depression' column).
    nullable_cols : list of str or None
        Columns that are allowed to contain NaN (e.g., 'Profession' which
        has NaN for students and will be handled in Feature Engineering).
        Defaults to `high_cardinality_cols` if None.

    Raises
    ------
    AssertionError
        Descriptive error for the first failing check.
    """
    if nullable_cols is None:
        nullable_cols = list(high_cardinality_cols)
    print("\n=== Validation Report ===")
    passed = 0
    failed = 0

    def _check(condition: bool, msg_pass: str, msg_fail: str):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {msg_pass}")
            passed += 1
        else:
            print(f"  FAIL  {msg_fail}")
            failed += 1

    # 1. Columns
    actual_cols = set(df.columns)
    expected_cols = set(expected_columns)
    _check(
        actual_cols == expected_cols,
        f"Columns match expected schema ({len(actual_cols)} cols).",
        f"Column mismatch. Extra: {actual_cols - expected_cols} | Missing: {expected_cols - actual_cols}",
    )

    # 2. No NaN (except explicitly allowed columns)
    null_counts = df.isnull().sum()
    unexpected_nan = {
        col: int(cnt)
        for col, cnt in null_counts.items()
        if cnt > 0 and col not in nullable_cols
    }
    _check(
        len(unexpected_nan) == 0,
        f"Zero NaN in non-nullable columns (allowed NaN in: {nullable_cols}).",
        f"Unexpected NaN found in: {unexpected_nan}",
    )

    # 3. High-cardinality columns are object/string dtype
    for col in high_cardinality_cols:
        if col in df.columns:
            # Pandas 3.0+ uses 'str' or 'string' for string columns, while older pandas uses object
            is_valid_type = df[col].dtype in (object, 'O', 'object', 'str', 'string')
            _check(
                is_valid_type,
                f"'{col}' is string dtype (kept for Target Encoding).",
                f"'{col}' has unexpected dtype: {df[col].dtype}",
            )

    # 4. Binary columns contain only 0/1
    binary_cols = [
        "Gender",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
        "Working Professional or Student",
        "has_cgpa",
    ]
    for col in binary_cols:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            _check(
                unique_vals <= {0, 1},
                f"'{col}' contains only {{0, 1}}.",
                f"'{col}' has unexpected values: {unique_vals}",
            )

    # 5. Ordinal columns in expected ranges
    if "Sleep Duration" in df.columns:
        unique_vals = set(df["Sleep Duration"].dropna().unique())
        _check(
            unique_vals <= {0, 1, 2, 3},
            "Sleep Duration values in {0, 1, 2, 3}.",
            f"Sleep Duration has unexpected values: {unique_vals}",
        )
    if "Dietary Habits" in df.columns:
        unique_vals = set(df["Dietary Habits"].dropna().unique())
        _check(
            unique_vals <= {0, 1, 2},
            "Dietary Habits values in {0, 1, 2}.",
            f"Dietary Habits has unexpected values: {unique_vals}",
        )

    # 6. Merged columns in valid numeric range
    for col in ["Pressure", "Satisfaction"]:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            _check(
                1.0 <= col_min and col_max <= 5.0,
                f"'{col}' values in [1.0, 5.0] (min={col_min:.1f}, max={col_max:.1f}).",
                f"'{col}' out of expected range: min={col_min}, max={col_max}",
            )

    # 7. Dropped columns are absent
    dropped_cols = [
        "id", "Name",
        "Academic Pressure", "Work Pressure",
        "Study Satisfaction", "Job Satisfaction",
    ]
    for col in dropped_cols:
        _check(
            col not in df.columns,
            f"'{col}' correctly absent.",
            f"'{col}' should have been dropped but is still present.",
        )

    # 8. Depression column unchanged (train only)
    if is_train and "Depression" in df.columns:
        val_counts = df["Depression"].value_counts()
        _check(
            set(df["Depression"].unique()) <= {0, 1},
            f"Depression column intact: {dict(val_counts)}.",
            f"Depression column has unexpected values: {df['Depression'].unique()}",
        )

    print(f"\nResult: {passed} passed, {failed} failed.")
    if failed > 0:
        raise AssertionError(f"{failed} validation check(s) failed. Review output above.")
    print("All checks passed. Data is ready to save.\n")


# ============================================================
# STAGE 8 — PERSISTENCE
# ============================================================

def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed",
    train_ids: pd.Series = None,
    test_ids: pd.Series = None,
    interim_dir: str = "data/interim",
) -> None:
    """
    Save processed train and test DataFrames to parquet and CSV.
    Optionally save ID series to the interim directory.

    Parquet is preferred (smaller, preserves dtypes exactly). If neither
    pyarrow nor fastparquet is installed the function falls back to CSV-only
    and prints a warning — the notebook continues without error.

    Parameters
    ----------
    train_df : pd.DataFrame
        Fully preprocessed training DataFrame.
    test_df : pd.DataFrame
        Fully preprocessed test DataFrame.
    output_dir : str
        Directory for processed parquet and CSV files.
    train_ids : pd.Series or None
        Training set IDs, saved for Kaggle submission construction in NB06.
    test_ids : pd.Series or None
        Test set IDs, saved for Kaggle submission construction in NB06.
    interim_dir : str
        Directory for intermediate files (IDs).

    Returns
    -------
    None
        Files are written to disk. Prints confirmation with sizes.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)

    # Detect parquet engine availability once
    _parquet_available = False
    try:
        import pyarrow  # noqa: F401
        _parquet_available = True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            _parquet_available = True
        except ImportError:
            print(
                "  [WARN] Neither pyarrow nor fastparquet is installed. "
                "Parquet files will be skipped — CSV only.\n"
                "  Install with:  pip install pyarrow"
            )

    files: dict = {}
    if _parquet_available:
        files[os.path.join(output_dir, "train_preprocessed.parquet")] = (train_df, "parquet")
        files[os.path.join(output_dir, "test_preprocessed.parquet")]  = (test_df,  "parquet")
    files[os.path.join(output_dir, "train_preprocessed.csv")] = (train_df, "csv")
    files[os.path.join(output_dir, "test_preprocessed.csv")]  = (test_df,  "csv")

    for path, (df, fmt) in files.items():
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved: {path}  ({size_kb:.1f} KB, shape={df.shape})")

    if train_ids is not None:
        path = os.path.join(interim_dir, "train_ids.csv")
        train_ids.to_csv(path, index=False, header=True)
        print(f"  Saved: {path}  ({len(train_ids)} IDs)")

    if test_ids is not None:
        path = os.path.join(interim_dir, "test_ids.csv")
        test_ids.to_csv(path, index=False, header=True)
        print(f"  Saved: {path}  ({len(test_ids)} IDs)")




# ============================================================
# ZONE: ĐÀM ĐẠT
# ============================================================
# Ghi chú: Các hàm Target Encoding, Feature Selection...

def cross_val_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target: str,
    n_splits: int = 5,
    smoothing: float = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply leakage-safe target encoding for one high-cardinality column.

    Train encoding is generated out-of-fold:
    each validation fold is transformed by an encoder fit only on the
    corresponding training folds. Test encoding is generated by an
    encoder fit on the full training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing both the feature column and target.
    test_df : pd.DataFrame
        Test data containing the feature column.
    col : str
        High-cardinality categorical column to encode.
    target : str
        Binary target column name.
    n_splits : int
        Number of folds for out-of-fold encoding.
    smoothing : float
        Smoothing strength passed to category_encoders.TargetEncoder.
    random_state : int
        Random seed for KFold shuffling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Encoded train column and encoded test column.
    """
    if col not in train_df.columns or col not in test_df.columns:
        raise KeyError(f"Column '{col}' must exist in both train and test.")
    if target not in train_df.columns:
        raise KeyError(f"Target column '{target}' not found in train_df.")
    from category_encoders import TargetEncoder

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_encoded = np.zeros(len(train_df), dtype=float)

    for tr_idx, val_idx in kf.split(train_df):
        encoder = TargetEncoder(cols=[col], smoothing=smoothing)
        encoder.fit(train_df.iloc[tr_idx][[col]], train_df.iloc[tr_idx][target])
        train_encoded[val_idx] = encoder.transform(
            train_df.iloc[val_idx][[col]]
        )[col].to_numpy()

    encoder_full = TargetEncoder(cols=[col], smoothing=smoothing)
    encoder_full.fit(train_df[[col]], train_df[target])
    test_encoded = encoder_full.transform(test_df[[col]])[col].to_numpy()

    return train_encoded, test_encoded


def add_high_cardinality_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    target: str,
    fill_value: str = "Unknown",
    n_splits: int = 5,
    smoothing: float = 300,
    drop_original: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target-encode a list of high-cardinality categorical columns safely.

    Missing values are replaced with `fill_value` so "missing" becomes a
    learnable category. Each encoded column is appended as `<col>_te`.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set containing the target.
    test_df : pd.DataFrame
        Test set without the target.
    cols : list[str]
        Columns to encode.
    target : str
        Target column name in train_df.
    fill_value : str
        Placeholder category used for NaN values.
    n_splits : int
        Number of folds used in out-of-fold encoding.
    smoothing : float
        Smoothing parameter for TargetEncoder.
    drop_original : bool
        If True, remove the original string columns after encoding.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Updated train and test DataFrames.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cols:
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)

        train_encoded, test_encoded = cross_val_target_encode(
            train_df=train_df,
            test_df=test_df,
            col=col,
            target=target,
            n_splits=n_splits,
            smoothing=smoothing,
        )
        encoded_col = f"{col}_te"
        train_df[encoded_col] = train_encoded
        test_df[encoded_col] = test_encoded

    if drop_original:
        train_df = train_df.drop(columns=cols)
        test_df = test_df.drop(columns=cols)

    return train_df, test_df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features motivated by domain knowledge and EDA.

    Returns a copied DataFrame with seven additional numeric features:
    stress_load, wellbeing_index, stress_satisfaction_ratio,
    hours_pressure_ratio, poor_sleep, cgpa_pressure, and risk_score.
    """
    df = df.copy()

    required_cols = [
        "Pressure",
        "Financial Stress",
        "Satisfaction",
        "Sleep Duration",
        "Work/Study Hours",
        "CGPA",
        "has_cgpa",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for interaction features: {missing}")

    df["stress_load"] = df["Pressure"] * df["Financial Stress"]
    df["wellbeing_index"] = (df["Satisfaction"] + df["Sleep Duration"]) / 2
    df["stress_satisfaction_ratio"] = df["Pressure"] / (df["Satisfaction"] + 1)
    df["hours_pressure_ratio"] = df["Work/Study Hours"] * df["Pressure"]
    df["poor_sleep"] = (df["Sleep Duration"] == 0).astype(int)
    df["cgpa_pressure"] = df["CGPA"] * df["Pressure"] * df["has_cgpa"]

    # Risk score: yeu to rui ro kep — ket hop suy nghi tu sat + tien su gia dinh
    # Gia tri: 0 = khong co yeu to rui ro, 1 = co 1 yeu to, 2 = ca hai yeu to
    df["risk_score"] = (
        df["Have you ever had suicidal thoughts ?"].astype(int)
        + df["Family History of Mental Illness"].astype(int)
    )

    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into four ordinal groups capturing non-linear risk patterns.

    Groups:
      0 = 18-29
      1 = 30-42
      2 = 43-51
      3 = 52+
    """
    if "Age" not in df.columns:
        raise KeyError("Column 'Age' not found in DataFrame.")

    df = df.copy()
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 29, 42, 51, 99],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    return df



# ============================================================
# ZONE: TÂM
# ============================================================
# Ghi chú: Các hàm tạo Interaction Features, Binning...
