"""
Post-hoc drift-detection analysis computed from block metrics (blocks_df)
and saved drift_points. Works only for models with a drift detector (IDDCW)
and datasets with known drift points.

Output (one row per Run_ID x Dataset x Model):
    Detection_Lag_samples     - distance from known_drift to first detection [samples]
                                                            NaN if model has no drift detector or drift did not occur
    Recovery_Time_samples     - number of samples until RWA_Score reaches
                                                            pre-drift level (baseline = median RWA from first third of stream)
                                                            NaN if no detection or no recovery occurred
    Pre_Drift_RWA             - median RWA in blocks before drift
    Post_Drift_Min_RWA        - minimum RWA in window after drift event
    RWA_Drop                  - Pre_Drift_RWA - Post_Drift_Min_RWA (drop magnitude)
    Stability_RWA_Std         - std RWA across all blocks (entire stream)
    Stability_RWA_IQR         - IQR RWA across all blocks
    Stability_RWA_Std_PostDrift - std RWA only in blocks after drift event

Implementation notes:
    - "Drift event" = known_drift_point (generator ground truth),
        not the detection itself. Detection Lag = detection - event.
    - Recovery is measured from the drift event, not from detection,
        to keep it comparable even for models without detector.
    - For gradual drift (Hyperplane, MC_Gradual), known_drift_point
        is the midpoint of the transition window.
    - For reoccurring drift (MC_Reoccurring), first drift event is used.
"""

import os
import numpy as np
import pandas as pd


# Known drift points (ground truth from generate_imbalanced_data.py)
# Value = sample index (after PRETRAIN_SIZE=2000 offset, see run_experiments_parallel.py) where drift occurs.
# For datasets without drift = None.
# For reoccurring = list of all drift points.
KNOWN_DRIFT_POINTS = {
    "SEA_Imb9010":               None,
    "Agrawal_Imb9010":           25000,
    "hyperplane_gradual_imb9010":None,   # continuous gradual drift - no point event
    "rbf_drift_balanced":        None,   # continuous gradual drift - no point event
    "MC_Abrupt_3C_70155":        25000,
    "MC_Gradual_3C_70155":       25000,  # transition midpoint
    "MC_Abrupt_4C_601555":       25000,
    "MC_Reoccurring_3C_80155":   [16667, 33333],  # 3-phase reoccurring
    "SEA_Balanced":              None,
    "Agrawal_Balanced":          None,
}

# Window before/after drift point for analysis [in blocks x BLOCK_SIZE samples]
PRE_DRIFT_WINDOW_BLOCKS  = 10   # 10 blocks = 5000 samples before drift
POST_DRIFT_WINDOW_BLOCKS = 20   # 20 blocks = 10000 samples after drift event


def _load_drift_points(preds_dir, d_name, m_name, run_id):
    """Load saved drift_points from worker .npz file."""
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
    Return Detection Lag = first detection after known_point - known_point.
    Consider only detections in window (known_point, known_point + max_lag].
    Returns NaN if no detection occurred in the window.
    """
    if not drift_points or known_point is None:
        return np.nan
    candidates = [d for d in drift_points if known_point < d <= known_point + max_lag]
    if not candidates:
        return np.nan
    return float(min(candidates) - known_point)


def _recovery_analysis(block_rwa, known_point, block_size, pre_window=10, post_window=20):
    """
    Analyze RWA drop and recovery around the drift event.

    Parameters
    ----------
    block_rwa    : pd.Series, index = Block_End (last sample index in block)
    known_point  : int, sample where drift occurred
    block_size   : int, block size (default 500)

    Returns
    -------
    dict with keys: pre_drift_rwa, post_drift_min_rwa, rwa_drop, recovery_time_samples
    """
    result = {
        "pre_drift_rwa":       np.nan,
        "post_drift_min_rwa":  np.nan,
        "rwa_drop":            np.nan,
        "recovery_time":       np.nan,
    }
    if known_point is None or len(block_rwa) == 0:
        return result

    # Blocks before and after drift event
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

    # Recovery time: first block after drift event where RWA >= 95% of pre-drift level
    recovery_threshold = pre_rwa * 0.95
    recovered = post_blocks[post_blocks >= recovery_threshold]
    if len(recovered) > 0:
        first_recovery_block = recovered.index[0]
        result["recovery_time"] = float(first_recovery_block - known_point)

    return result


def compute_drift_stats(blocks_df, preds_dir, block_size=500):
    """
    Compute post-hoc drift-related statistics from block metrics.

    Parameters
    ----------
    blocks_df  : pd.DataFrame, prequential_block_metrics.csv
    preds_dir  : str, directory with prediction .npz files
    block_size : int, block size (default 500)

    Returns
    -------
    pd.DataFrame with one row per (Run_ID, Dataset, Model)
    """
    rows = []

    for (run_id, d_name, m_name), grp in blocks_df.groupby(["Run_ID", "Dataset", "Model"]):
        grp = grp.sort_values("Block_End")
        block_rwa = grp.set_index("Block_End")["RWA_Score"]

        # Stability metrics (entire stream)
        stability_std  = float(block_rwa.std())  if len(block_rwa) > 1 else np.nan
        q75, q25       = block_rwa.quantile(0.75), block_rwa.quantile(0.25)
        stability_iqr  = float(q75 - q25)

        # Drift-specific analysis
        known = KNOWN_DRIFT_POINTS.get(d_name)

        # For reoccurring drift - use the first drift point
        if isinstance(known, list):
            primary_known = known[0]
        else:
            primary_known = known

        drift_points = _load_drift_points(preds_dir, d_name, m_name, run_id)

        det_lag = _detection_lag(drift_points, primary_known)

        rec = _recovery_analysis(block_rwa, primary_known, block_size,
                                 PRE_DRIFT_WINDOW_BLOCKS, POST_DRIFT_WINDOW_BLOCKS)

        # Stability only in post-drift window
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