"""
visualize_datasets.py
─────────────────────
Vizualizácia datasetov pre diplomovú prácu — vyznačenie driftu a nevyváženosti.
Štýl inšpirovaný Figure 3 z Sarnovský & Kolařík (2021), PeerJ CS.

Pre každý dataset generuje 2 grafy:
  1. Feature importance v čase (Gini importance z Online Random Forest)
     — scatter plot kde veľkosť bodky = dôležitosť príznaku v danom chunke
     — zmeny v dôležitosti naznačujú feature drift
  2. Distribúcia tried v čase (stacked area / bar chart)
     — ukazuje priebeh imbalance ratio a zmeny v distribúcii tried

Known drift pointy sú vyznačené červenými vertikálnymi čiarami.

Vstupy:
  - CSV datasety z ./data/synthetic_imbalanced/ a ./data/synthetic_multiclass/
  - CSV datasety z ./data/real/real_clean/

Výstupy (do results/figures/):
  - dataset_drift_<name>.png       — combined figure (feature imp + class dist)

Spustenie:
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

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR    = "./data/"
FIGURES_DIR = "./results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300
CHUNK_SIZE = 1000   # veľkosť chunkov pre analýzu

# Datasety s cestami
DATASETS = {
    # Syntetické binárne
    "SEA_Imb9010":                "synthetic_imbalanced/sea_abrupt_imb9010.csv",
    "Agrawal_Imb9010":            "synthetic_imbalanced/agrawal_drift_imb9010.csv",
    "hyperplane_gradual_imb9010": "synthetic_imbalanced/hyperplane_gradual_imb9010.csv",
    "rbf_drift_imb9010":          "synthetic_imbalanced/rbf_drift_imb9010.csv",
    # Syntetické multiclass
    "MC_Abrupt_3C_70155":         "synthetic_multiclass/mc_abrupt_3c_70155.csv",
    "MC_Gradual_3C_70155":        "synthetic_multiclass/mc_gradual_3c_70155.csv",
    "MC_Abrupt_4C_601555":        "synthetic_multiclass/mc_abrupt_4c_601555.csv",
    "MC_Reoccurring_3C_80155":    "synthetic_multiclass/mc_reoccurring_3c_80155.csv",
    # Reálne
    "ELEC":      "real/real_clean/elec_clean.csv",
    "KDD99":     "real/real_clean/kdd99_clean.csv",
    "Airlines":  "real/real_clean/airlines_clean.csv",
    "Shuttle":   "real/real_clean/shuttle_clean.csv",
    "CoverType": "real/real_clean/covtype_clean.csv",
    "Jigsaw":    "real/real_clean/jigsaw_clean.csv",
    "ElectCovid": "real/real_clean/elect_covid_clean.csv",
    "FakeNewsComb": "real/real_clean/comb_clean.csv",
}

# Known drift pointy (ground truth)
KNOWN_DRIFTS = {
    "SEA_Imb9010":                None,
    "Agrawal_Imb9010":            [25000],
    "hyperplane_gradual_imb9010": None,      # kontinuálny
    "rbf_drift_imb9010":          None,      # kontinuálny
    "MC_Abrupt_3C_70155":         [25000],
    "MC_Gradual_3C_70155":        [25000],
    "MC_Abrupt_4C_601555":        [25000],
    "MC_Reoccurring_3C_80155":    [16667, 33333],
    # Reálne — neznáme, ale vyznačíme feature drift
    "ELEC": None, "KDD99": None, "Airlines": None,
    "Shuttle": None, "CoverType": None, "Jigsaw": None,
    "ElectCovid": None, "FakeNewsComb": None,

}

# Akademický štýl
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})


# ═══════════════════════════════════════════════════════════════════════════════
# NAČÍTANIE DÁT
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(name, path):
    """Načíta dataset, vráti (X, y) alebo None."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE V ČASE  (Figure 3 štýl z článku)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance_over_time(X, y, chunk_size=1000, max_features_show=15):
    """
    Vypočíta feature importance pomocou jednoduchého Random Forest
    trénovaného na posuvných chunkoch. Vracia DataFrame s importance
    per feature per chunk.

    Pre veľmi veľké datasety (>100k) sa vzorkuje na max 100k.
    """
    from sklearn.ensemble import RandomForestClassifier

    n = len(y)
    n_features = X.shape[1]

    # Obmedzenie pre veľké datasety
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

        # Aspoň 2 triedy v chunke
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


def plot_feature_importance(ax, imp_df, dataset_name, drift_points=None):
    """
    Scatter plot: x = pozícia v streame, y = index príznaku, veľkosť = importance.
    Štýl: Figure 3 z PeerJ článku.
    """
    if imp_df.empty:
        ax.text(0.5, 0.5, "nedostatok dát", transform=ax.transAxes, ha="center")
        return

    # Veľkosť bodiek — normalizovaná na viditeľnosť
    max_imp = imp_df["importance"].max()
    if max_imp > 0:
        sizes = (imp_df["importance"] / max_imp) * 80 + 2
    else:
        sizes = 5

    # Farba podľa príznaku (colormap)
    scatter = ax.scatter(
        imp_df["chunk_center"], imp_df["feature"],
        s=sizes, c=imp_df["feature"],
        cmap="viridis", alpha=0.7, edgecolors="none",
    )

    # Drift čiary
    if drift_points:
        for dp in drift_points:
            ax.axvline(x=dp, color="#e74c3c", linewidth=1.5, linestyle="--",
                       alpha=0.8, zorder=10)

    ax.set_xlabel("Samples")
    ax.set_ylabel("Feature index")
    ax.set_title(f"({dataset_name}) Feature importance", fontweight="bold", fontsize=9)

    # Y-os: celočíselné indexy
    n_feat = int(imp_df["feature"].max()) + 1
    if n_feat <= 20:
        ax.set_yticks(range(n_feat))
    else:
        ax.set_yticks(range(0, n_feat, max(1, n_feat // 10)))


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBÚCIA TRIED V ČASE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_distribution(y, chunk_size=1000):
    """Vypočíta distribúciu tried per chunk."""
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


def plot_class_distribution(ax, dist_df, classes, dataset_name, drift_points=None):
    """
    Stacked area chart — distribúcia tried v čase.
    Zmeny v rozdelení vizuálne naznačujú drift.
    """
    if dist_df.empty:
        ax.text(0.5, 0.5, "nedostatok dát", transform=ax.transAxes, ha="center")
        return

    x = dist_df["chunk_center"].values
    cols = [f"class_{c}" for c in classes]

    # Farebná paleta
    cmap = plt.cm.Set2 if len(classes) <= 8 else plt.cm.tab20
    colors = [cmap(i / max(1, len(classes) - 1)) for i in range(len(classes))]

    # Stacked area
    bottom = np.zeros(len(x))
    for i, col in enumerate(cols):
        vals = dist_df[col].values
        ax.fill_between(x, bottom, bottom + vals, alpha=0.7,
                        color=colors[i], label=f"Trieda {classes[i]}")
        bottom += vals

    # Drift čiary
    if drift_points:
        for dp in drift_points:
            ax.axvline(x=dp, color="#e74c3c", linewidth=1.5, linestyle="--",
                       alpha=0.8, zorder=10, label="Drift" if dp == drift_points[0] else "")

    ax.set_xlabel("Samples")
    ax.set_ylabel("Podiel tried")
    ax.set_title(f"({dataset_name}) Distribúcia tried", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=6, ncol=min(4, len(classes) + 1))


# ═══════════════════════════════════════════════════════════════════════════════
# KOMBINOVANÝ GRAF PRE JEDEN DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset_figure(name, X, y, drift_points, filename):
    """Generuje kombinovaný obrázok: feature importance + distribúcia tried."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5),
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    # Feature importance
    print(f"    Computing feature importance...", end="", flush=True)
    imp_df = compute_feature_importance_over_time(X, y, chunk_size=CHUNK_SIZE)
    print(f" done ({len(imp_df)} points)")
    plot_feature_importance(ax1, imp_df, name, drift_points)

    # Distribúcia tried
    dist_df, classes = compute_class_distribution(y, chunk_size=CHUNK_SIZE)
    plot_class_distribution(ax2, dist_df, classes, name, drift_points)

    fig.tight_layout(h_pad=2.5)

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  ✓  {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRID — viacero datasetov na jednej strane (len distribúcia tried)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_class_dist_grid(loaded, category, filename):
    """Grid distribúcií tried pre kategóriu datasetov (syntetické/reálne)."""
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
        dp = KNOWN_DRIFTS.get(name)
        if isinstance(dp, int):
            dp = [dp]
        plot_class_distribution(ax, dist_df, classes, name, dp)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Distribúcia tried v čase — {category}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1.0])

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  ✓  {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Vizualizácia datasetov — drift a nevyváženosť")
    print("  Štýl: Figure 3 z Sarnovský & Kolařík (2021)")
    print("=" * 65)

    loaded_synth = {}
    loaded_real = {}

    print("\nNačítavam datasety...")
    for name, path in DATASETS.items():
        X, y = load_dataset(name, path)
        if X is not None:
            cat = "synth" if "synthetic" in path else "real"
            print(f"  OK  {name:<35} {X.shape[0]:>7} × {X.shape[1]:>3}  triedy={sorted(set(y))}")
            if cat == "synth":
                loaded_synth[name] = (X, y)
            else:
                loaded_real[name] = (X, y)
        else:
            print(f"  --  {name:<35} nenájdený")

    # ── Individuálne grafy (feature importance + class distribution) ───────
    print("\n── Individuálne grafy (feature imp + class dist) ──")
    all_loaded = {**loaded_synth, **loaded_real}
    for name, (X, y) in all_loaded.items():
        dp = KNOWN_DRIFTS.get(name)
        if isinstance(dp, int):
            dp = [dp]
        safe = name.lower().replace(" ", "_")
        print(f"  {name}:")
        generate_dataset_figure(name, X, y, dp, f"dataset_drift_{safe}.png")

    # ── Grid grafy (len distribúcia tried) ────────────────────────────────
    print("\n── Grid: distribúcia tried ──")
    if loaded_synth:
        generate_class_dist_grid(loaded_synth, "Syntetické datasety",
                                 "class_dist_grid_synth.png")
    if loaded_real:
        generate_class_dist_grid(loaded_real, "Reálne datasety",
                                 "class_dist_grid_real.png")

    print(f"\n{'=' * 65}")
    print(f"  Hotovo. → {os.path.abspath(FIGURES_DIR)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()