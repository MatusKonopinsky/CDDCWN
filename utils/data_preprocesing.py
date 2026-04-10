"""
utils/data_preprocesing.py
──────────────────────────
Načítavanie a predspracovanie datasetov pre streamové experimenty.

Exportované funkcie:
  read_clean_csv(filename)       — načítanie predspracovaného CSV (používa runner)
  preprocess_all_real_datasets() — jednorazové predspracovanie raw reálnych datasetov

Reálne datasety sa predspracujú raz príkazom:
    python -m utils.data_preprocesing

Výstup: ./data/real/real_clean/<dataset>_clean.csv (MinMax škálované, bez hlavičky,
        posledný stĺpec = integer label)

Očakávané raw súbory v ./data/real/real_raw/:
    elec.csv                              — Electricity
    kddcup.data_10_percent_corrected      — KDD99 (alebo kdd99.csv)
    airlines.csv                          — Airlines
    covtype.data / covtype.csv            — Forest CoverType
    shuttle.trn + shuttle.tst             — Shuttle (alebo shuttle.csv)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

RAW_DIR  = "./data/real/real_raw/"
OUT_DIR  = "./data/real/real_clean/"


# ══════════════════════════════════════════════════════════════════════════════
# FUNKCIA POUŽÍVANÁ RUNNEROM
# ══════════════════════════════════════════════════════════════════════════════

def read_clean_csv(filename):
    """
    Načíta predspracovaný CSV súbor (bez hlavičky, posledný stĺpec = label).
    Používa pandas namiesto np.loadtxt — rádovo rýchlejšie na veľkých datasetoch
    (napr. CovType 581k, KDD99 494k).
    Vracia (data, X, y) ako numpy polia.
    """
    # Autodetekcia hlavičky — ak prvý riadok nie je číselný, preskočíme ho
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


# ══════════════════════════════════════════════════════════════════════════════
# INTERNÉ POMOCNÉ FUNKCIE PRE PREDSPRACOVANIE
# ══════════════════════════════════════════════════════════════════════════════

def _save_clean(X: np.ndarray, y: np.ndarray, name: str):
    """MinMax škáluje X a uloží dataset do CSV bez hlavičky."""
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)
    data = np.concatenate([X_sc, y.reshape(-1, 1)], axis=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_clean.csv")
    np.savetxt(out_path, data, delimiter=",", fmt="%.8f")
    print(f"  ✓  {name:<20} {X.shape[0]:>7} vzoriek  "
          f"{X.shape[1]:>2} príznakov  "
          f"triedy={sorted(set(y.astype(int)))}  → {out_path}")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-enkóduje všetky object stĺpce v DataFrame."""
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PREDSPRACOVANIE JEDNOTLIVÝCH DATASETOV
# ══════════════════════════════════════════════════════════════════════════════

def _process_elec():
    """
    Electricity — 45 312 vzoriek, 6 príznakov, 2 triedy (~42/58).
    Stĺpec 'date' je identifikátor, zahadzujeme ho.
    Label: UP=1 / DOWN=0.
    """
    path = os.path.join(RAW_DIR, "elec.csv")
    if not os.path.exists(path):
        print(f"  SKIP elec — nenájdený: {path}")
        return
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    _save_clean(X, y, "elec")


def _process_kdd99():
    """
    KDD Cup 99 — 10% verzia, ~494 021 vzoriek, 41 príznakov.
    Konvertujeme na binárny problém: normal=0, attack=1.
    3 kategorické stĺpce (protocol_type, service, flag) sú label-enkódované.
    Niektoré verzie súboru majú bodku na konci labelu — automaticky ju odoberieme.
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
        print(f"  SKIP kdd99 — nenájdený v {RAW_DIR}")
        print(f"             skúšané: {candidates}")
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

    # Zisti či súbor má hlavičku
    first_val = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
    has_header = isinstance(first_val, str) and first_val == "duration"
    df = pd.read_csv(path, names=col_names, header=0 if has_header else None)
    df.dropna(inplace=True)

    X = _encode_categoricals(df.iloc[:, :-1].copy()).values.astype(float)

    # Binárny label: normal=0, attack=1 (bodka na konci niektorých verzií)
    raw_labels = df["label"].astype(str).str.rstrip(".")
    y = (raw_labels != "normal").astype(int).values
    _save_clean(X, y, "kdd99")


def _process_airlines():
    """
    Airlines — 539 383 vzoriek, 7 príznakov, 2 triedy (delay/no-delay).
    Kategorické: Airline, AirportFrom, AirportTo.
    """
    path = os.path.join(RAW_DIR, "airlines.csv")
    if not os.path.exists(path):
        print(f"  SKIP airlines — nenájdený: {path}")
        return
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = _encode_categoricals(df)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    _save_clean(X, y, "airlines")


def _process_covtype():
    """
    Forest CoverType — 581 012 vzoriek, 54 príznakov, 7 tried.
    Niektoré verzie majú hlavičku (Elevation,...), iné nie.
    Label je 1-based (1–7) → prekonvertujeme na 0-based (0–6).
    """
    candidates = ["covtype.data", "covtype.csv", "covtype.data.gz"]
    path = next((os.path.join(RAW_DIR, c) for c in candidates
                 if os.path.exists(os.path.join(RAW_DIR, c))), None)
    if path is None:
        print(f"  SKIP covtype — nenájdený v {RAW_DIR}")
        return

    # Autodetekcia hlavičky
    try:
        first = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
        has_header = isinstance(first, str) and not first.replace(".", "").lstrip("-").isdigit()
    except Exception:
        has_header = False

    df = pd.read_csv(path, header=0 if has_header else None, low_memory=False)
    df.dropna(inplace=True)
    df = _encode_categoricals(df)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    if y.min() == 1:
        y = y - 1  # 1-based → 0-based
    _save_clean(X, y, "covtype")


def _process_shuttle():
    """
    Shuttle — ~58 000 vzoriek, 9 príznakov, 7 tried (~80% trieda 0).
    Spájame shuttle.trn + shuttle.tst ak existujú, inak shuttle.csv.
    Prvý stĺpec je čas/identifikátor — zahadzujeme ho.
    Label je 1-based (1–7) → 0-based (0–6).
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
                # shuttle.csv z arff2csv má hlavičku
                df_tmp = pd.read_csv(p)
                # Zahod prípadný prvý stĺpec ak je to čas (numerický, monotónny)
                dfs.append(df_tmp)
                break

    if not dfs:
        print(f"  SKIP shuttle — nenájdený v {RAW_DIR}")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)
    df = _encode_categoricals(df)

    # Ak má prvý stĺpec monotónne rastúce hodnoty → je to index/čas, zahodíme
    first_col = df.iloc[:, 0].values.astype(float)
    is_time_col = np.all(np.diff(first_col[:1000]) >= 0) if len(first_col) > 1000 else False
    if is_time_col:
        X = df.iloc[:, 1:-1].values.astype(float)
    else:
        X = df.iloc[:, :-1].values.astype(float)

    y = df.iloc[:, -1].values.astype(int)
    if y.min() == 1:
        y = y - 1  # 1-based → 0-based
    _save_clean(X, y, "shuttle")


# ══════════════════════════════════════════════════════════════════════════════
# HLAVNÁ FUNKCIA — volá sa jednorazovo pred experimentmi
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_all_real_datasets():
    """
    Predspracuje všetky dostupné reálne datasety z ./data/real/.
    Datasety ktoré nenájde, preskočí s varovaním.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nPreprocessing reálnych datasetov z: {os.path.abspath(RAW_DIR)}")
    print("─" * 70)
    _process_elec()
    _process_kdd99()
    _process_airlines()
    _process_covtype()
    _process_shuttle()
    print("─" * 70)
    print("Hotovo.\n")


if __name__ == "__main__":
    preprocess_all_real_datasets()