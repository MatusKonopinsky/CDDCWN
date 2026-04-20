"""
Dataset loading and preprocessing for streaming experiments.

Exported functions:
    read_clean_csv(filename)       - load preprocessed CSV (used by runner)
    preprocess_all_real_datasets() - one-time preprocessing of raw real datasets

Real datasets are preprocessed once with:
        python -m utils.data_preprocesing

Output: ./data/real/real_clean/<dataset>_clean.csv (MinMax scaled, no header,
                last column = integer label)

Expected raw files in ./data/real/real_raw/:
        elec.csv                              - Electricity
        kddcup.data_10_percent_corrected      - KDD99 (or kdd99.csv)
        airlines.csv                          - Airlines
        shuttle.trn + shuttle.tst             - Shuttle (or shuttle.csv)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RAW_DIR  = "./data/real/real_raw/"
OUT_DIR  = "./data/real/real_clean/"


# =============================================================================
# FUNCTION USED BY THE RUNNER
# =============================================================================
def read_clean_csv(filename):
    """
    Load preprocessed CSV (no header, last column = label).
    Uses pandas instead of np.loadtxt - significantly faster on large datasets
    (e.g. CovType 581k, KDD99 494k).
    Returns (data, X, y) as numpy arrays.
    """
    # Header auto-detection - skip first row if it is not numeric
    try:
        first = pd.read_csv(filename, nrows=1, header=None).iloc[0, 0]
        has_header = isinstance(first, str) and not first.replace(".", "").lstrip("-").isdigit()
    except Exception:
        has_header = False

    df = pd.read_csv(filename, header=0 if has_header else None, dtype=float)
    data_np = df.values

    if data_np.ndim == 1:
        data_np = data_np.reshape(1, -1)

    X_np = data_np[:, :-1]
    y_np = data_np[:, -1].astype(int)

    return data_np, X_np, y_np


# =============================================================================
# INTERNAL PREPROCESSING HELPERS
# =============================================================================
def _save_clean(X: np.ndarray, y: np.ndarray, name: str):
    """MinMax-scale X and save dataset to CSV without header."""
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)
    data = np.concatenate([X_sc, y.reshape(-1, 1)], axis=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_clean.csv")
    np.savetxt(out_path, data, delimiter=",", fmt="%.8f")
    print(f"    {name:<20} {X.shape[0]:>7} samples  "
          f"{X.shape[1]:>2} features  "
          f"classes={sorted(set(y.astype(int)))}  -> {out_path}")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns in a DataFrame."""
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# =============================================================================
# PREPROCESSING OF INDIVIDUAL DATASETS
# =============================================================================
def _process_elec():
    """
    Electricity - 45,312 samples, 6 features, 2 classes (~42/58).
    Column 'date' is an identifier and is dropped.
    Label: UP=1 / DOWN=0.
    """
    path = os.path.join(RAW_DIR, "elec.csv")
    if not os.path.exists(path):
        print(f"  SKIP elec - not found: {path}")
        return
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    _save_clean(X, y, "elec")


def _process_kdd99():
    """
    KDD Cup 99 - 10% version, ~494,021 samples, 41 features.
    Converted to a binary problem: normal=0, attack=1.
    3 categorical columns (protocol_type, service, flag) are label-encoded.
    Some file variants have a trailing period in labels - removed automatically.
    """
    candidates = [
        "kddcup.data_10_percent_corrected",
        "kddcup.data_10_percent_corrected_new",
        "kddcup.data_10_percent",
        "kdd99.csv",
        "kddcup99.csv",
    ]
    path = next((os.path.join(RAW_DIR, c) for c in candidates
                 if os.path.exists(os.path.join(RAW_DIR, c))), None)
    if path is None:
        print(f"  SKIP kdd99 - not found in {RAW_DIR}")
        print(f"             tried: {candidates}")
        return

    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","label",
    ]

    # Detect whether the file has a header
    first_val = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
    has_header = isinstance(first_val, str) and first_val == "duration"
    df = pd.read_csv(path, names=col_names, header=0 if has_header else None)
    df.dropna(inplace=True)

    X = _encode_categoricals(df.iloc[:, :-1].copy()).values.astype(float)

    # Binary label: normal=0, attack=1 (some variants have trailing period)
    raw_labels = df["label"].astype(str).str.rstrip(".")
    y = (raw_labels != "normal").astype(int).values
    _save_clean(X, y, "kdd99")


def _process_airlines():
    """
    Airlines - 539,383 samples, 7 features, 2 classes (delay/no-delay).
    Categorical: Airline, AirportFrom, AirportTo.
    """
    path = os.path.join(RAW_DIR, "airlines.csv")
    if not os.path.exists(path):
        print(f"  SKIP airlines - not found: {path}")
        return
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = _encode_categoricals(df)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    _save_clean(X, y, "airlines")

def _process_shuttle():
    """
    Shuttle - ~58,000 samples, 9 features, 7 classes (~80% class 0).
    Merge shuttle.trn + shuttle.tst when available, otherwise shuttle.csv.
    First column is time/identifier and is dropped.
    Label is 1-based (1-7) -> converted to 0-based (0-6).
    """
    dfs = []
    for fname in ["shuttle.trn", "shuttle.tst"]:
        p = os.path.join(RAW_DIR, fname)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, header=None, sep=r"\s+"))

    if not dfs:
        for fname in ["shuttle.csv", "shuttle.data"]:
            p = os.path.join(RAW_DIR, fname)
            if os.path.exists(p):
                # shuttle.csv from arff2csv has a header
                df_tmp = pd.read_csv(p)
                # Drop possible first column if it is time (numeric, monotonic)
                dfs.append(df_tmp)
                break

    if not dfs:
        print(f"  SKIP shuttle - not found in {RAW_DIR}")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)
    df = _encode_categoricals(df)

    # If first column is monotonically increasing -> treat as index/time and drop
    first_col = df.iloc[:, 0].values.astype(float)
    is_time_col = np.all(np.diff(first_col[:1000]) >= 0) if len(first_col) > 1000 else False
    if is_time_col:
        X = df.iloc[:, 1:-1].values.astype(float)
    else:
        X = df.iloc[:, :-1].values.astype(float)

    y = df.iloc[:, -1].values.astype(int)
    if y.min() == 1:
        y = y - 1  # 1-based -> 0-based
    _save_clean(X, y, "shuttle")


# =============================================================================
# MAIN FUNCTION - run once before experiments
# =============================================================================
def preprocess_all_real_datasets():
    """
    Preprocess all available real datasets from ./data/real/.
    Datasets not found are skipped with a warning.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nPreprocessing real datasets from: {os.path.abspath(RAW_DIR)}")
    print("─" * 70)
    _process_elec()
    _process_kdd99()
    _process_airlines()
    _process_shuttle()
    print("─" * 70)
    print("Done.\n")


if __name__ == "__main__":
    preprocess_all_real_datasets()