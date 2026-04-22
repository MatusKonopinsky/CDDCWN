"""
Preprocess balanced RBF Drift dataset provided by the supervisor.

Input:  ./data/real/real_raw/rbf_drift.csv
        - ~1,000,000 samples, 10 features + 'class' column
        - String labels: 'class1', 'class2'
        - Class ratio ~50/50 (balanced)
        - Feature ranges: roughly [-2, 3]

Pipeline:
    1. Load CSV with header
    2. Label-encode 'class1' -> 0, 'class2' -> 1
    3. Truncate to TARGET_TOTAL samples (keeps temporal ordering)
    4. MinMax scale features to [0, 1]
    5. Save as CSV

Output: ./data/synthetic_imbalanced/rbf_drift_balanced.csv
        - TARGET_TOTAL samples, 10 features + integer class label
        - Class ratio ~50/50 (balanced, drift preserved)

Run:
    python preprocess_rbf.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


INPUT_PATH = "./data/synthetic_raw/rbf_drift.csv"
OUTPUT_PATH = "./data/synthetic_imbalanced/rbf_drift_balanced.csv"

# Set to None to keep the full dataset (~1M samples).
# Set to e.g. 50_000 to match the length of other binary synthetic datasets.
TARGET_TOTAL = None

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def sanity_check(X, y, n_chunks=10):
    n = len(y)
    print(f"\n--- Sanity check ---")
    print(f"Total length: {n}")
    classes, counts = np.unique(y, return_counts=True)
    global_ratio = {int(c): cnt / n for c, cnt in zip(classes, counts)}
    print(f"Global ratio: {global_ratio}")

    chunk = n // n_chunks
    for i in range(n_chunks):
        s = i * chunk
        e = s + chunk if i < n_chunks - 1 else n
        yc = y[s:e]
        classes, counts = np.unique(yc, return_counts=True)
        ratios = {int(c): cnt / len(yc) for c, cnt in zip(classes, counts)}
        print(f"Chunk {i+1}/{n_chunks} ({s}-{e}): "
              f"class0={ratios.get(0, 0):.3f}, class1={ratios.get(1, 0):.3f}")


def main():
    # 1) Load
    print(f"Loading from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        return
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {len(df)} samples, {df.shape[1] - 1} features")

    # 2) Extract features + labels, label-encode classes
    X = df.iloc[:, :-1].values.astype(float)
    y_raw = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"  Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 3) Truncate to TARGET_TOTAL if requested (keeps temporal order)
    if TARGET_TOTAL is not None and len(y) > TARGET_TOTAL:
        print(f"\nTruncating to first {TARGET_TOTAL} samples (drift preserved)...")
        X = X[:TARGET_TOTAL]
        y = y[:TARGET_TOTAL]

    # 4) MinMax scale
    print("Scaling features (MinMax to [0, 1])...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) Save with header, matching the format of other synthetic datasets
    print(f"Saving to: {OUTPUT_PATH}")
    df_out = pd.DataFrame(X_scaled, columns=[f"attr_{i}" for i in range(X_scaled.shape[1])])
    df_out["class"] = y.astype(int)
    df_out.to_csv(OUTPUT_PATH, index=False)

    # 6) Sanity check
    sanity_check(X_scaled, y, n_chunks=10)


if __name__ == "__main__":
    main()