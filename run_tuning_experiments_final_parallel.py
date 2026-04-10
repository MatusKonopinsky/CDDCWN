"""
run_experiments.py — Runner pre streamové experimenty s DDCW a baselinami.

Využíva:
  utils/logger.py        — live dashboard, queue komunikácia
  utils/metrics.py       — výpočet RWA, F1, Kappa, AUC, minority metrík
  utils/model_factory.py — definícia modelov a get_model_configs
  utils/data_preprocesing.py — načítanie CSV datasetov
  utils/rwa_metric.py    — RWA implementácia
  model/configurable_ddcw_new.py — DDCW model
"""

import os
import sys
import copy
import time
import pickle
import warnings
import multiprocessing

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from skmultiflow.data.data_stream import DataStream

from utils.data_preprocesing import read_clean_csv
from utils.logger        import _logger_process, worker_init, log
from utils.metrics       import compute_main_metrics
from utils.model_factory import get_model_name, get_model_configs
from utils.drift_metrics import compute_drift_stats


# ============================================================
# KONFIGURÁCIA
# ============================================================

NUMBER_OF_RUNS = 5       # zvýš na 5 pre finálnu štatistickú validáciu

DATA_DIR    = "./data/"
RESULTS_DIR = "./results/"

# Počet vzoriek zo začiatku streamu použitých na inicializáciu modelu PRED
# prequential hodnotením. Tieto vzorky sa nezarátavajú do metrík.
# Hodnota 2000 bola overená ablačným testom — zlepšuje studený štart DDCW
# na Agrawal a RBF bez regrésie na ostatných datasetoch.
PRETRAIN_SIZE = 2000

BLOCK_SIZE    = 500

SYNTHETIC_DATASETS = {
    # Binárne (imbalance 90/10)
    "SEA_Imb9010":               "synthetic_imbalanced/sea_abrupt_imb9010.csv",
    "Agrawal_Imb9010":           "synthetic_imbalanced/agrawal_drift_imb9010.csv",
    "hyperplane_gradual_imb9010":"synthetic_imbalanced/hyperplane_gradual_imb9010.csv",
    "rbf_drift_imb9010":         "synthetic_imbalanced/rbf_drift_imb9010.csv",
    # Multiclass
    "MC_Abrupt_3C_70155":        "synthetic_multiclass/mc_abrupt_3c_70155.csv",
    "MC_Gradual_3C_70155":       "synthetic_multiclass/mc_gradual_3c_70155.csv",
    "MC_Abrupt_4C_601555":       "synthetic_multiclass/mc_abrupt_4c_601555.csv",
    "MC_Reoccurring_3C_80155":   "synthetic_multiclass/mc_reoccurring_3c_80155.csv",
}

# Reálne datasety — výstupy z preprocess_real_datasets.py
# Všetky sú vo formáte MinMaxScaled CSV bez hlavičky, posledný stĺpec = label.
# Ak niektorý _clean.csv neexistuje, dataset sa preskočí s varovaním.
REAL_DATASETS = {
    # Binárne
    #"ELEC":     "real/real_clean/elec_clean.csv",      # 45 312 vzoriek, 6 príznakov, 2 triedy (~42/58)
    #"KDD99":    "real/real_clean/kdd99_clean.csv",      # 494 021 vzoriek, 41 príznakov, 2 triedy (normal/attack)
    #"Airlines": "real/real_clean/airlines_clean.csv",   # 539 383 vzoriek, 7 príznakov, 2 triedy
    # Multiclass
    #"Shuttle":  "real/real_clean/shuttle_clean.csv",    # 58 000 vzoriek, 9 príznakov, 7 tried (~80% trieda 0)
    # Text
    #"Jigsaw":  "real/real_clean/jigsaw_clean.csv",
    "ElectCovid": "real/real_clean/elect_covid_clean.csv",
    "FakeNewsComb": "real/real_clean/comb_clean.csv",
}

# "synthetic"  — len syntetické datasety
# "real"       — len reálne datasety
# "all"        — všetky datasety
DATASET_MODE = "real"


# ============================================================
# WORKER
# ============================================================

def _run_one_dataset(args):
    """
    Worker: spracuje jeden (run_id, d_name) pár sekvenčne cez všetky modely.
    Prijíma cestu k súboru namiesto numpy polí — vyhýba sa serializácii
    veľkých polí (CovType 250MB, KDD99 160MB) cez spawn IPC na Windowse.
    _LOG_QUEUE je nastavená cez worker_init, nie cez task tuple (Windows/spawn fix).
    """
    run_id, d_name, d_path, preds_dir = args

    from utils.data_preprocesing import read_clean_csv as _read
    try:
        _, X_data, y_data = _read(d_path)
    except Exception as e:
        log("error", run_id, d_name, "load", f"Načítanie zlyhalo: {e}")
        return [], []

    import warnings as _w
    _w.filterwarnings("ignore")

    result_rows = []
    block_rows  = []

    try:
        stream = DataStream(X_data.copy(), y_data.copy())
    except Exception as e:
        log("error", run_id, d_name, "DataStream", str(e))
        return result_rows, block_rows

    for base_model in get_model_configs(run_id=run_id):
        model  = copy.deepcopy(base_model)
        m_name, params = get_model_name(model)

        log("start", run_id, d_name, m_name)
        stream.restart()

        y_true_list       = []
        y_pred_list       = []
        y_proba_full_list = []
        block_y_true      = []
        block_y_pred      = []
        block_y_proba     = []
        model_sizes       = []
        n_samples         = 0
        start_time        = time.time()

        try:
            if hasattr(model, "reset"):
                model.reset()

            # ── Pretrain ───────────────────────────────────────────────────
            if stream.n_remaining_samples() >= PRETRAIN_SIZE:
                X_pre, y_pre = stream.next_sample(PRETRAIN_SIZE)
                model.partial_fit(X_pre, y_pre, classes=stream.target_values)

            n_total_classes  = stream.n_classes
            total_to_process = stream.n_remaining_samples()

            # ── Prequential loop ───────────────────────────────────────────
            while stream.has_more_samples():
                X_s, y_s = stream.next_sample()

                # Predikcia
                try:
                    proba = model.predict_proba(X_s)
                    pred  = int(np.argmax(proba[0]))
                except Exception:
                    proba = None
                    pred  = None

                if proba is None:
                    try:
                        pred = int(model.predict(X_s)[0])
                    except Exception:
                        pred = 0
                    proba = np.zeros((1, n_total_classes))
                    proba[0, pred] = 1.0

                # Normalizácia pravdepodobností na celý rozsah tried
                full_proba = np.zeros(n_total_classes)
                cp = np.asarray(proba[0])
                full_proba[:len(cp)] = cp[:n_total_classes]
                pred = int(np.clip(pred, 0, n_total_classes - 1))

                y_true_list.append(int(y_s[0]))
                y_pred_list.append(pred)
                y_proba_full_list.append(full_proba)
                block_y_true.append(int(y_s[0]))
                block_y_pred.append(pred)
                block_y_proba.append(full_proba)

                model.partial_fit(X_s, y_s)
                n_samples += 1

                # Meranie veľkosti modelu každých 200 vzoriek
                if n_samples % 200 == 0:
                    try:
                        model_sizes.append(sys.getsizeof(pickle.dumps(model)))
                    except Exception:
                        pass

                # Progress správa každých 5000 vzoriek
                if n_samples % 5000 == 0:
                    log("progress", run_id, d_name, m_name, n_samples, total_to_process)

                # Uloženie blokových metrík každých BLOCK_SIZE vzoriek
                if len(block_y_true) >= BLOCK_SIZE:
                    bm = compute_main_metrics(
                        y_true=np.array(block_y_true,   dtype=int),
                        y_pred=np.array(block_y_pred,   dtype=int),
                        y_proba=np.array(block_y_proba, dtype=float),
                        n_total_classes=n_total_classes,
                    )
                    block_rows.append({
                        "Run_ID": run_id, "Dataset": d_name,
                        "Model": m_name, "Block_End": n_samples, **bm,
                    })
                    block_y_true, block_y_pred, block_y_proba = [], [], []

            # Posledný neúplný blok
            if block_y_true:
                bm = compute_main_metrics(
                    y_true=np.array(block_y_true,   dtype=int),
                    y_pred=np.array(block_y_pred,   dtype=int),
                    y_proba=np.array(block_y_proba, dtype=float),
                    n_total_classes=n_total_classes,
                )
                block_rows.append({
                    "Run_ID": run_id, "Dataset": d_name,
                    "Model": m_name, "Block_End": n_samples, **bm,
                })

            log("progress", run_id, d_name, m_name, total_to_process, total_to_process)
            total_time    = time.time() - start_time
            avg_update_ts = (np.mean(model.update_times)
                             if hasattr(model, "update_times") and model.update_times else 0.0)
            avg_mem_kb    = (np.mean(model_sizes) / 1024.0) if model_sizes else 0.0

            y_true_arr  = np.array(y_true_list,      dtype=int)
            y_pred_arr  = np.array(y_pred_list,       dtype=int)
            y_proba_arr = np.array(y_proba_full_list, dtype=float)

            # ── Finálne metriky ────────────────────────────────────────────
            metrics = compute_main_metrics(
                y_true=y_true_arr, y_pred=y_pred_arr,
                y_proba=y_proba_arr, n_total_classes=n_total_classes,
            )

            # ── Uloženie predikcií ─────────────────────────────────────────
            drift_points = list(getattr(model, "_drift_points", []))
            save_dict = {"y_true": y_true_arr, "y_pred": y_pred_arr, "y_proba": y_proba_arr}
            if drift_points:
                save_dict["drift_points"] = np.array(drift_points, dtype=int)
            np.savez_compressed(
                os.path.join(preds_dir, f"{d_name}_{m_name[:30].replace(chr(47), chr(95))}_run{run_id}.npz"),
                **save_dict,
            )

            result_row = {
                "Run_ID": run_id, "Dataset": d_name, "Model": m_name,
                "Total_Samples_Evaluated": int(len(y_true_arr)),
                **metrics,
                "Total_Time_s":      total_time,
                "Avg_Update_Time_s": avg_update_ts,
                "Memory_kB":         avg_mem_kb,
                "Drift_Detections":  len(drift_points),
            }
            result_row.update(params)
            result_rows.append(result_row)

            log("done", run_id, d_name, m_name, {
                "RWA_Score":            metrics.get("RWA_Score",            -1),
                "G_Mean":               metrics.get("G_Mean",               -1),
                "Mean_Minority_Recall": metrics.get("Mean_Minority_Recall", -1),
                "Drift_Detections":     len(drift_points),
            }, total_time)

        except Exception as e:
            log("error", run_id, d_name, m_name, str(e))
            result_rows.append({
                "Run_ID": run_id, "Dataset": d_name, "Model": m_name,
                "RWA_Score": -1.0, "Macro_F1": -1.0,
                "Mean_Minority_F1": -1.0, "Worst_Minority_F1": -1.0,
                "Drift_Detections": 0,
            })

    log("task_done", run_id, d_name)
    return result_rows, block_rows


# ============================================================
# HLAVNÝ LOOP
# ============================================================

def run_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    preds_dir = os.path.normpath(os.path.join(RESULTS_DIR, "predictions"))
    os.makedirs(preds_dir, exist_ok=True)

    # ── Načítanie datasetov ────────────────────────────────────────────────
    print("Nahrávam datasety...")
    loaded_datasets = {}
    dataset_paths   = {}  # d_name -> absolútna cesta k _clean.csv

    # Výber datasetov podľa DATASET_MODE
    if DATASET_MODE == "synthetic":
        all_datasets = dict(SYNTHETIC_DATASETS)
    elif DATASET_MODE == "real":
        all_datasets = dict(REAL_DATASETS)
    elif DATASET_MODE == "all":
        all_datasets = {**SYNTHETIC_DATASETS, **REAL_DATASETS}
    else:
        raise ValueError(f"Neznámy DATASET_MODE: {DATASET_MODE!r}. Použi 'synthetic', 'real' alebo 'all'.")

    print(f"  Režim: {DATASET_MODE!r}  ({len(all_datasets)} datasetov)")

    for d_name, d_filename in all_datasets.items():
        file_path = os.path.join(DATA_DIR, d_filename)
        if not os.path.exists(file_path):
            print(f"  VAROVANIE: {file_path} neexistuje, preskočené.")
            continue
        try:
            _, X, y = read_clean_csv(file_path)
            if X.shape[0] == 0:
                print(f"  VAROVANIE: {d_name} je prázdny po načítaní, preskočené.")
                continue
            loaded_datasets[d_name] = (X, y)
            dataset_paths[d_name]   = os.path.abspath(file_path)
            print(f"  OK  {d_name:<35} {X.shape[0]:>7} vzoriek  "
                  f"{X.shape[1]:>2} príznakov  triedy={sorted(set(y.astype(int)))}")
        except Exception as e:
            print(f"  CHYBA pri načítaní {d_name}: {e}")

    total_tasks = len(dataset_paths) * NUMBER_OF_RUNS
    n_workers   = min(multiprocessing.cpu_count(), total_tasks)

    print(f"\n{'═' * 80}")
    print(f"  Datasety: {len(loaded_datasets)}  |  Behy: {NUMBER_OF_RUNS}  |  "
          f"Úlohy: {total_tasks}  |  Workery: {n_workers}")
    print(f"{'═' * 80}\n")

    # ── Spustenie logger procesu ────────────────────────────────────────────
    ctx = multiprocessing.get_context("spawn")
    log_queue   = ctx.Queue()
    logger_proc = ctx.Process(
        target=_logger_process,
        args=(log_queue, total_tasks),
        daemon=True,
    )
    logger_proc.start()

    # ── Zostavenie úloh ────────────────────────────────────────────────────
    # Odovzdávame cestu k súboru namiesto numpy polí — vyhneme sa serializácii
    # veľkých polí (CovType 250MB, KDD99 160MB) cez spawn IPC na Windowse.
    # Worker načíta dáta sám — každý worker má vlastnú kópiu v pamäti.
    tasks = [
        (run_id, d_name, d_path, preds_dir)
        for run_id in range(1, NUMBER_OF_RUNS + 1)
        for d_name, d_path in dataset_paths.items()
    ]

    all_results    = []
    all_block_rows = []

    with ctx.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(log_queue,),
    ) as pool:
        for result_rows, block_rows in pool.imap_unordered(_run_one_dataset, tasks):
            all_results.extend(result_rows)
            all_block_rows.extend(block_rows)

    log_queue.put(("STOP",))
    logger_proc.join(timeout=10)

    # ── Uloženie výsledkov ─────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("Ukladám výsledky...")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "grid_search_results_raw.csv"),
        index=False, float_format="%.6f",
    )
    print(f"  Raw výsledky:     {RESULTS_DIR}grid_search_results_raw.csv")

    blocks_df = pd.DataFrame(all_block_rows)
    blocks_df.to_csv(
        os.path.join(RESULTS_DIR, "prequential_block_metrics.csv"),
        index=False, float_format="%.6f",
    )
    print(f"  Blokové metriky:  {RESULTS_DIR}prequential_block_metrics.csv")

    # ── Drift + Stability štatistiky (post-hoc z blokových metrík) ────────
    try:
        drift_stats_df = compute_drift_stats(blocks_df, preds_dir, block_size=BLOCK_SIZE)
        drift_stats_df.to_csv(
            os.path.join(RESULTS_DIR, "drift_stability_stats.csv"),
            index=False, float_format="%.6f",
        )
        print(f"  Drift/Stability:  {RESULTS_DIR}drift_stability_stats.csv")
    except Exception as e:
        print(f"  VAROVANIE: drift_stats zlyhali: {e}")

    summary = results_df.groupby(["Dataset", "Model"]).agg(
        Avg_RWA=                  ("RWA_Score",             "mean"),
        Std_RWA=                  ("RWA_Score",             "std"),
        Avg_G_Mean=               ("G_Mean",                "mean"),
        Std_G_Mean=               ("G_Mean",                "std"),
        Avg_Macro_F1=             ("Macro_F1",              "mean"),
        Std_Macro_F1=             ("Macro_F1",              "std"),
        Avg_Weighted_F1=          ("Weighted_F1",           "mean"),
        Avg_Mean_Minority_F1=     ("Mean_Minority_F1",      "mean"),
        Std_Mean_Minority_F1=     ("Mean_Minority_F1",      "std"),
        Avg_Worst_Minority_F1=    ("Worst_Minority_F1",     "mean"),
        Avg_Mean_Minority_Recall= ("Mean_Minority_Recall",  "mean"),
        Std_Mean_Minority_Recall= ("Mean_Minority_Recall",  "std"),
        Avg_Worst_Minority_Recall=("Worst_Minority_Recall", "mean"),
        Avg_Majority_Recall=      ("Majority_Recall",       "mean"),
        Avg_Kappa=                ("Kappa",                 "mean"),
        Avg_AUC=                  ("AUC",                   "mean"),
        Avg_Time=                 ("Total_Time_s",          "mean"),
        Avg_Memory=               ("Memory_kB",             "mean"),
        Avg_Drift_Detections=     ("Drift_Detections",      "mean"),
        Std_Drift_Detections=     ("Drift_Detections",      "std"),
    ).reset_index()

    summary.to_csv(
        os.path.join(RESULTS_DIR, "grid_search_summary.csv"),
        index=False, float_format="%.6f",
    )
    print(f"  Súhrnné výsledky: {RESULTS_DIR}grid_search_summary.csv")
    print(f"{'═' * 80}\n")
    print(summary.sort_values(["Dataset", "Avg_RWA"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    run_experiments()