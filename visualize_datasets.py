"""
Visualize dataset dynamics for drift and class-imbalance inspection.

For each dataset, this script creates:
    - feature-importance-over-time view
    - class-distribution-over-time view

Outputs are written to:
    - results/figures/

Run:
    python visualize_datasets.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR    = "./data/"
FIGURES_DIR = "./results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300
CHUNK_SIZE = 1000   # chunk size for analysis

# Datasets and paths
DATASETS = {
    # Synthetic binary
    "SEA_Imb9010":                "synthetic_imbalanced/sea_abrupt_imb9010.csv",
    "Agrawal_Imb9010":            "synthetic_imbalanced/agrawal_drift_imb9010.csv",
    "hyperplane_gradual_imb9010": "synthetic_imbalanced/hyperplane_gradual_imb9010.csv",
    "rbf_drift_imb9010":          "synthetic_imbalanced/rbf_drift_imb9010.csv",
    "SEA_Bal":                "synthetic_imbalanced/sea_balanced.csv",
    "Agrawal_Bal":            "synthetic_imbalanced/agrawal_balanced.csv",
    # Synthetic multiclass
    "MC_Abrupt_3C_70155":         "synthetic_multiclass/mc_abrupt_3c_70155.csv",
    "MC_Gradual_3C_70155":        "synthetic_multiclass/mc_gradual_3c_70155.csv",
    "MC_Abrupt_4C_601555":        "synthetic_multiclass/mc_abrupt_4c_601555.csv",
    "MC_Reoccurring_3C_80155":    "synthetic_multiclass/mc_reoccurring_3c_80155.csv",
    # Real
    "ELEC":      "real/real_clean/elec_clean.csv",
    "KDD99":     "real/real_clean/kdd99_clean.csv",
    "Airlines":  "real/real_clean/airlines_clean.csv",
    "Shuttle":   "real/real_clean/shuttle_clean.csv",
    "Jigsaw":    "real/real_clean/jigsaw_clean.csv",
    "ElectCovid": "real/real_clean/elect_covid_clean.csv",
    "FakeNewsComb": "real/real_clean/comb_clean.csv",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})


# =============================================================================
# DATA LOADING
# =============================================================================
def load_dataset(name, path):
    """Load dataset and return (X, y) or None."""
    full = os.path.join(DATA_DIR, path)
    if not os.path.exists(full):
        return None, None

    try:
        first = pd.read_csv(full, nrows=1, header=None).iloc[0, 0]
        has_hdr = isinstance(first, str) and not first.replace(".", "").lstrip("-").isdigit()
    except Exception:
        has_hdr = False

    df = pd.read_csv(full, header=0 if has_hdr else None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y


# =============================================================================
# FEATURE IMPORTANCE OVER TIME
# =============================================================================
def compute_feature_importance_over_time(X, y, chunk_size=1000, max_features_show=15):
    """
    Compute feature importance using a simple Random Forest
    trained on sliding chunks. Returns a DataFrame with
    per-feature, per-chunk importance.

    For very large datasets (>100k), data is sampled down to 100k.
    """
    from sklearn.ensemble import RandomForestClassifier

    n = len(y)
    n_features = X.shape[1]

    # Limit for large datasets
    if n > 100000:
        idx = np.linspace(0, n - 1, 100000, dtype=int)
        X = X[idx]
        y = y[idx]
        n = len(y)

    rows = []
    for start in range(0, n - chunk_size + 1, chunk_size):
        end = start + chunk_size
        Xc = X[start:end]
        yc = y[start:end]

        # At least 2 classes in a chunk
        if len(np.unique(yc)) < 2:
            continue

        try:
            rf = RandomForestClassifier(n_estimators=10, max_depth=5,
                                        random_state=42, n_jobs=1)
            rf.fit(Xc, yc)
            imp = rf.feature_importances_
        except Exception:
            imp = np.zeros(n_features)

        for fi in range(min(n_features, max_features_show)):
            rows.append({
                "chunk_center": start + chunk_size // 2,
                "feature": fi,
                "importance": imp[fi] if fi < len(imp) else 0.0,
            })

    return pd.DataFrame(rows)


def plot_feature_importance(ax, imp_df, dataset_name):
    if imp_df.empty:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes, ha="center")
        return

    # Pivot to matrix: rows = features, columns = chunks
    pivot = imp_df.pivot(index="feature", columns="chunk_center", values="importance")
    matrix = pivot.values  # shape: (n_features, n_chunks)

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
    )

    # X-axis: labels based on actual stream positions
    chunk_centers = pivot.columns.values
    n_ticks = min(6, len(chunk_centers))
    tick_idx = np.linspace(0, len(chunk_centers) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(chunk_centers[i] / 1000)}k" for i in tick_idx])

    # Y-axis: feature indices
    n_feat = len(pivot.index)
    if n_feat <= 20:
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(pivot.index)
    else:
        step = max(1, n_feat // 10)
        ax.set_yticks(range(0, n_feat, step))
        ax.set_yticklabels(pivot.index[::step])

    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Importance")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Feature index")
    ax.set_title(f"{dataset_name} — feature importance", fontweight="bold", fontsize=9)


# =============================================================================
# CLASS DISTRIBUTION OVER TIME
# =============================================================================
def compute_class_distribution(y, chunk_size=1000):
    """Compute class distribution per chunk."""
    n = len(y)
    classes = sorted(np.unique(y))
    rows = []

    for start in range(0, n - chunk_size + 1, chunk_size):
        end = start + chunk_size
        yc = y[start:end]
        counts = Counter(yc)
        total = len(yc)
        row = {"chunk_center": start + chunk_size // 2}
        for c in classes:
            row[f"class_{c}"] = counts.get(c, 0) / total
        rows.append(row)

    return pd.DataFrame(rows), classes


def plot_class_distribution(ax, dist_df, classes, dataset_name):
    if dist_df.empty:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes, ha="center")
        return

    x = dist_df["chunk_center"].values
    cols = [f"class_{c}" for c in classes]

    # Color palette - tab10/tab20 for strong, distinguishable colors
    if len(classes) <= 10:
        base_colors = list(plt.cm.tab10.colors)
    elif len(classes) <= 20:
        base_colors = list(plt.cm.tab20.colors)
    else:
        base_colors = [plt.cm.hsv(i / len(classes)) for i in range(len(classes))]
    colors = [base_colors[i % len(base_colors)] for i in range(len(classes))]

    # Stacked area + boundary line for better readability
    bottom = np.zeros(len(x))
    for i, col in enumerate(cols):
        vals = dist_df[col].values
        ax.fill_between(x, bottom, bottom + vals, alpha=0.82,
                color=colors[i], label=f"Class {classes[i]}")
        ax.plot(x, bottom + vals, color=colors[i], linewidth=0.9, alpha=0.95)
        bottom += vals

    ax.set_xlabel("Samples")
    ax.set_ylabel("Class proportion")
    ax.set_title(dataset_name, fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.0)
    # Legenda mimo subplotu, napravo — nezakryje priebeh
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6,
        frameon=False,
        title="Class",
        title_fontsize=7,
    )


# =============================================================================
# COMBINED PLOT FOR A SINGLE DATASET
# =============================================================================
def generate_dataset_figure(name, X, y, filename):
    """Generate a combined figure: feature importance + class distribution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    # Feature importance
    print(f"    Computing feature importance...", end="", flush=True)
    imp_df = compute_feature_importance_over_time(X, y, chunk_size=CHUNK_SIZE)
    print(f" done ({len(imp_df)} points)")
    plot_feature_importance(ax1, imp_df, name)

    # Class distribution
    dist_df, classes = compute_class_distribution(y, chunk_size=CHUNK_SIZE)
    plot_class_distribution(ax2, dist_df, classes, name)

    fig.tight_layout(h_pad=2.5)

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"    {filename}")


# =============================================================================
# GRID - multiple datasets on one page (class distribution only)
# =============================================================================
def generate_class_dist_grid(loaded, category, filename):
    """Class-distribution grid for a dataset category (synthetic/real)."""
    items = [(name, X, y) for name, (X, y) in loaded.items()
             if X is not None]

    if not items:
        return

    n = len(items)
    ncols = 2 if n <= 4 else 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 2.8),
                             squeeze=False)

    for idx, (name, X, y) in enumerate(items):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        dist_df, classes = compute_class_distribution(y, chunk_size=CHUNK_SIZE)
        plot_class_distribution(ax, dist_df, classes, name)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    if "synth" in category.lower() or "Synthetic" in category:
        title = "Class distribution over time — Synthetic datasets"
    else:
        title = "Class distribution over time — Real datasets"

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.97, 1.0])

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"    {filename}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  Dataset visualization - drift and imbalance")
    print("=" * 65)

    loaded_synth = {}
    loaded_real = {}

    print("\nLoading datasets...")
    for name, path in DATASETS.items():
        X, y = load_dataset(name, path)
        if X is not None:
            cat = "synth" if "synthetic" in path else "real"
            print(f"  OK  {name:<35} {X.shape[0]:>7} x {X.shape[1]:>3}  classes={sorted(set(y))}")
            if cat == "synth":
                loaded_synth[name] = (X, y)
            else:
                loaded_real[name] = (X, y)
        else:
            print(f"  --  {name:<35} not found")

    # Individual plots (feature importance + class distribution)
    print("\n── Individual plots (feature imp + class dist) ──")
    all_loaded = {**loaded_synth, **loaded_real}
    for name, (X, y) in all_loaded.items():
        safe = name.lower().replace(" ", "_")
        print(f"  {name}:")
        generate_dataset_figure(name, X, y, f"dataset_drift_{safe}.png")

    # Grid plots (class distribution only)
    print("\n── Grid: class distribution ──")
    if loaded_synth:
        generate_class_dist_grid(loaded_synth, "Synthetic datasets",
                                "class_dist_grid_synth.png")
    if loaded_real:
        generate_class_dist_grid(loaded_real, "Real datasets",
                                "class_dist_grid_real.png")

    print(f"\n{'=' * 65}")
    print(f"  Done. Output saved to: {os.path.abspath(FIGURES_DIR)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()