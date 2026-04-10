"""
utils/drift_metrics.py — Post-hoc analýza drift detekcie.

Počíta sa z blokových metrík (blocks_df) a uložených drift_points.
Funguje len pre modely s drift detektorom (DDCW) a datasety
s known drift pointami.

Exportované funkcie:
  compute_drift_stats(blocks_df, preds_dir, known_drift_points) -> pd.DataFrame

Výstup (jeden riadok per Run_ID × Dataset × Model):
  Detection_Lag_samples     — vzdialenosť od known_drift ku prvej detekcii [vzorky]
                              NaN ak model nemá drift detektor alebo drift nenastal
  Recovery_Time_samples     — počet vzoriek kým RWA_Score dosiahne pred-driftovú
                              úroveň (baseline = median RWA z prvej tretiny streamu)
                              NaN ak nebola detekcia alebo nedošlo k zotaveniu
  Pre_Drift_RWA             — medián RWA v blokoch pred driftom
  Post_Drift_Min_RWA        — minimum RWA v okne po driftovom evente
  RWA_Drop                  — Pre_Drift_RWA − Post_Drift_Min_RWA  (veľkosť pádu)
  Stability_RWA_Std         — std RWA cez všetky bloky (celý stream)
  Stability_RWA_IQR         — IQR RWA cez všetky bloky
  Stability_RWA_Std_PostDrift — std RWA len v blokoch po driftovom evente

Poznámky k implementácii:
  - "Driftový event" = known_drift_point (ground truth z generátora),
    nie detekcia samotná. Detection Lag = detekcia − event.
  - Recovery sa počíta od driftového eventu, nie od detekcie,
    aby bol porovnateľný aj medzi modelmi bez detektora.
  - Pre gradual drift (Hyperplane, MC_Gradual) je known_drift_point
    stred prechodového okna.
  - Pre reoccurring drift (MC_Reoccurring) sa berie prvý drift event.
"""

import os
import numpy as np
import pandas as pd


# ── Known drift pointy (ground truth z generate_imbalanced_data.py) ──────────
# Hodnota = číslo vzorky (po pretrain_size=500 offset) kde nastáva drift.
# Pre datasety bez driftu = None.
# Pre reoccurring = zoznam všetkých drift pointov.
KNOWN_DRIFT_POINTS = {
    "SEA_Imb9010":               None,
    "Agrawal_Imb9010":           25000,
    "hyperplane_gradual_imb9010":None,   # kontinuálny gradual — žiadny bodový event
    "rbf_drift_imb9010":         None,   # kontinuálny gradual — žiadny bodový event
    "MC_Abrupt_3C_70155":        25000,
    "MC_Gradual_3C_70155":       25000,  # stred prechodu
    "MC_Abrupt_4C_601555":       25000,
    "MC_Reoccurring_3C_80155":   [16667, 33333],  # 3-fázový reoccurring
}

# Okno pred/po drift pointe pre analýzu [v blokoch × BLOCK_SIZE vzoriek]
PRE_DRIFT_WINDOW_BLOCKS  = 10   # 10 blokov = 5000 vzoriek pred driftom
POST_DRIFT_WINDOW_BLOCKS = 20   # 20 blokov = 10000 vzoriek po driftovom evente


def _load_drift_points(preds_dir, d_name, m_name, run_id):
    """Načíta uložené drift_points z .npz súboru workera."""
    path = os.path.normpath(os.path.join(preds_dir, f"{d_name}_{m_name[:30].replace(chr(47), chr(95))}_run{run_id}.npz"))
    if not os.path.exists(path):
        return []
    try:
        npz = np.load(path, allow_pickle=True)
        if "drift_points" in npz:
            return list(npz["drift_points"].astype(int))
    except Exception:
        pass
    return []


def _detection_lag(drift_points, known_point, max_lag=5000):
    """
    Vráti Detection Lag = prvá detekcia po known_point − known_point.
    Uvažujeme len detekcie v okne (known_point, known_point + max_lag].
    Vracia NaN ak žiadna detekcia v okne nenastala.
    """
    if not drift_points or known_point is None:
        return np.nan
    candidates = [d for d in drift_points if known_point < d <= known_point + max_lag]
    if not candidates:
        return np.nan
    return float(min(candidates) - known_point)


def _recovery_analysis(block_rwa, known_point, block_size, pre_window=10, post_window=20):
    """
    Analyzuje pád a zotavenie RWA okolo driftového eventu.

    Parametre
    ----------
    block_rwa    : pd.Series, index = Block_End (číslo poslednej vzorky bloku)
    known_point  : int, vzorka kde nastal drift
    block_size   : int, veľkosť bloku (default 500)

    Vracia
    ------
    dict s kľúčmi: pre_drift_rwa, post_drift_min_rwa, rwa_drop, recovery_time_samples
    """
    result = {
        "pre_drift_rwa":       np.nan,
        "post_drift_min_rwa":  np.nan,
        "rwa_drop":            np.nan,
        "recovery_time":       np.nan,
    }
    if known_point is None or len(block_rwa) == 0:
        return result

    # Bloky pred a po driftovom evente
    pre_blocks  = block_rwa[block_rwa.index <= known_point].tail(pre_window)
    post_blocks = block_rwa[block_rwa.index >  known_point].head(post_window)

    if len(pre_blocks) == 0:
        return result

    pre_rwa = float(pre_blocks.median())
    result["pre_drift_rwa"] = pre_rwa

    if len(post_blocks) == 0:
        return result

    result["post_drift_min_rwa"] = float(post_blocks.min())
    result["rwa_drop"] = max(0.0, pre_rwa - result["post_drift_min_rwa"])

    # Recovery time: prvý blok po driftovom evente kde RWA >= 95% pred-driftovej úrovne
    recovery_threshold = pre_rwa * 0.95
    recovered = post_blocks[post_blocks >= recovery_threshold]
    if len(recovered) > 0:
        first_recovery_block = recovered.index[0]
        result["recovery_time"] = float(first_recovery_block - known_point)

    return result


def compute_drift_stats(blocks_df, preds_dir, block_size=500):
    """
    Vypočíta drift-related štatistiky post-hoc z blokových metrík.

    Parametre
    ----------
    blocks_df  : pd.DataFrame, prequential_block_metrics.csv
    preds_dir  : str, adresár s .npz súbormi predikcií
    block_size : int, veľkosť bloku (default 500)

    Vracia
    ------
    pd.DataFrame s jedným riadkom per (Run_ID, Dataset, Model)
    """
    rows = []

    for (run_id, d_name, m_name), grp in blocks_df.groupby(["Run_ID", "Dataset", "Model"]):
        grp = grp.sort_values("Block_End")
        block_rwa = grp.set_index("Block_End")["RWA_Score"]

        # ── Stability metriky (celý stream) ──────────────────────────────
        stability_std  = float(block_rwa.std())  if len(block_rwa) > 1 else np.nan
        q75, q25       = block_rwa.quantile(0.75), block_rwa.quantile(0.25)
        stability_iqr  = float(q75 - q25)

        # ── Drift-specific analýza ────────────────────────────────────────
        known = KNOWN_DRIFT_POINTS.get(d_name)

        # Pre reoccurring — použijeme prvý drift point
        if isinstance(known, list):
            primary_known = known[0]
        else:
            primary_known = known

        drift_points = _load_drift_points(preds_dir, d_name, m_name, run_id)

        det_lag = _detection_lag(drift_points, primary_known)

        rec = _recovery_analysis(block_rwa, primary_known, block_size,
                                 PRE_DRIFT_WINDOW_BLOCKS, POST_DRIFT_WINDOW_BLOCKS)

        # Stability len v post-drift okne
        if primary_known is not None:
            post_blocks = block_rwa[block_rwa.index > primary_known].head(POST_DRIFT_WINDOW_BLOCKS)
            stability_std_post = float(post_blocks.std()) if len(post_blocks) > 1 else np.nan
        else:
            stability_std_post = np.nan

        rows.append({
            "Run_ID":                    run_id,
            "Dataset":                   d_name,
            "Model":                     m_name,
            "Detection_Lag_samples":     det_lag,
            "Recovery_Time_samples":     rec["recovery_time"],
            "Pre_Drift_RWA":             rec["pre_drift_rwa"],
            "Post_Drift_Min_RWA":        rec["post_drift_min_rwa"],
            "RWA_Drop":                  rec["rwa_drop"],
            "Stability_RWA_Std":         stability_std,
            "Stability_RWA_IQR":         stability_iqr,
            "Stability_RWA_Std_PostDrift": stability_std_post,
        })

    return pd.DataFrame(rows)