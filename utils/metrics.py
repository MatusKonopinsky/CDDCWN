"""
utils/metrics.py — Výpočet hodnotiacich metrík pre streamové experimenty.

Exportované funkcie:
  safe_auc(y_true, y_proba, n_total_classes) -> float
  compute_main_metrics(y_true, y_pred, y_proba, n_total_classes) -> dict
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

from utils.rwa_metric import calculate_rwa


def safe_auc(y_true, y_proba, n_total_classes):
    """
    Bezpečný výpočet ROC-AUC — vracia 0.5 pri chybe alebo jednej triede.
    Pre multiclass používa OvR stratégiu.
    """
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.5
    try:
        if n_total_classes > 2:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    except Exception:
        return 0.5


def compute_main_metrics(y_true, y_pred, y_proba, n_total_classes):
    """
    Vypočíta všetky hodnodiace metriky pre jeden beh / blok.

    Parametre
    ----------
    y_true          : array-like int, skutočné triedy
    y_pred          : array-like int, predikované triedy
    y_proba         : array (n_samples, n_total_classes), pravdepodobnosti tried
    n_total_classes : int, celkový počet tried v datasete

    Vracia
    ------
    dict s kľúčmi:
      Accuracy, Weighted_F1, Macro_F1, Kappa, AUC, RWA_Score,
      G_Mean, Majority_Class, Majority_Recall, Minority_Classes,
      Mean_Minority_F1, Worst_Minority_F1,
      Mean_Minority_Recall, Worst_Minority_Recall
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_pred  = np.asarray(y_pred,  dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    labels_all = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    counts = pd.Series(y_true).value_counts().sort_index()

    majority_class   = int(counts.idxmax())
    minority_classes = [int(c) for c in counts.index if int(c) != majority_class]

    accuracy    = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    kappa       = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    rwa         = float(calculate_rwa(y_true, y_pred, np.arange(n_total_classes)))
    auc_score   = float(safe_auc(y_true, y_proba, n_total_classes))

    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels_all,
        average=None,
        zero_division=0,
    )

    df_cls = pd.DataFrame({
        "class":     [int(c)          for c in labels_all],
        "precision": [float(p)        for p in prec_arr],
        "recall":    [float(r)        for r in rec_arr],
        "f1":        [float(f)        for f in f1_arr],
        "support":   [int(s)          for s in supp_arr],
    })
    df_min = df_cls[df_cls["class"] != majority_class]

    # ── Minority / majority štatistiky ────────────────────────────────────
    maj_row = df_cls[df_cls["class"] == majority_class]
    majority_recall = float(maj_row["recall"].values[0]) if len(maj_row) > 0 else 0.0

    if len(df_min) > 0:
        mean_minority_f1      = float(df_min["f1"].mean())
        worst_minority_f1     = float(df_min["f1"].min())
        mean_minority_recall  = float(df_min["recall"].mean())
        worst_minority_recall = float(df_min["recall"].min())
    else:
        mean_minority_f1 = worst_minority_f1 = 0.0
        mean_minority_recall = worst_minority_recall = 0.0

    # ── G-Mean (Geometric Mean of per-class recalls) ───────────────────────
    # Štandard imbalanced learning literatúry (Kubat & Matwin 1997).
    # Pre multiclass = geometrický priemer recallov VŠETKÝCH tried.
    # Pre binárny = √(Recall_majority × Recall_minority).
    # Ak je recall akejkoľvek triedy = 0, G-Mean = 0.
    all_recalls = [majority_recall] + list(df_min["recall"].values) if len(df_min) > 0 else [majority_recall]
    if all(r > 0 for r in all_recalls):
        g_mean = float(np.prod(all_recalls) ** (1.0 / len(all_recalls)))
    else:
        g_mean = 0.0

    return {
        "Accuracy":             accuracy,
        "Weighted_F1":          weighted_f1,
        "Macro_F1":             macro_f1,
        "Kappa":                kappa,
        "AUC":                  auc_score,
        "RWA_Score":            rwa,
        "G_Mean":               g_mean,
        "Majority_Class":       majority_class,
        "Majority_Recall":      majority_recall,
        "Minority_Classes":     ",".join(map(str, minority_classes)),
        "Mean_Minority_F1":     mean_minority_f1,
        "Worst_Minority_F1":    worst_minority_f1,
        "Mean_Minority_Recall": mean_minority_recall,
        "Worst_Minority_Recall":worst_minority_recall,
    }