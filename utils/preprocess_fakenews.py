"""
preprocess_fakenews.py
──────────────────────
Predspracuje Fake News datasety pre streamové experimenty.

Datasety:
  1. elect_covid.csv — detekcia falošných príspevkov, simulovaná zmena tém
     z "election" na "covid" (concept drift cez zmenu domény)
  2. comb.csv — multitopic verzia s viacerými zmenami tém

Oba datasety majú rovnakú štruktúru: topic, text, label
  - topic: téma príspevku (ignoruje sa — slúži len na simuláciu driftu)
  - text: text príspevku
  - label: 0 = reálny, 1 = falošný (fake)
  - Poradie riadkov simuluje stream (bez timestampu, ale zoradené)

Pipeline:
  1. Načítanie CSV, ignorovanie stĺpca 'topic'
  2. Čistenie textu (lowercase, odstránenie URL a špeciálnych znakov)
  3. Načítanie pretrained embedding modelu (gensim.downloader)
  4. Výpočet TF-IDF váh na celom korpuse
  5. Reprezentácia dokumentu = TF-IDF vážený priemer pretrained vektorov
  6. MinMax škálovanie + uloženie ako clean CSV

Pretrained model:
  - "glove-wiki-gigaword-100" — 400k slov, 100d, Wikipedia + Gigaword
    Vhodnejší pre novinárske/politické texty ako Twitter-based model.
    Fake news datasety obsahujú formálnejší jazyk ako toxic komentáre.

Vstup:  ./data/real/real_text/elect_covid.csv
        ./data/real/real_text/comb.csv
Výstup: ./data/real/real_clean/elect_covid_clean.csv
        ./data/real/real_clean/comb_clean.csv

Spustenie:
    python preprocess_fakenews.py
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict


# ── Parametre ─────────────────────────────────────────────────────────────────
DATA_DIR = "./data/real/real_text/"
OUT_DIR = "./data/real/real_clean/"

DATASETS = {
    "elect_covid": {
        "raw": os.path.join(DATA_DIR, "elect_covid.csv"),
        "out": os.path.join(OUT_DIR, "elect_covid_clean.csv"),
        "desc": "Election→COVID concept drift (2 témy)",
    },
    "comb": {
        "raw": os.path.join(DATA_DIR, "comb.csv"),
        "out": os.path.join(OUT_DIR, "comb_clean.csv"),
        "desc": "Multi-topic concept drift (viacero tém)",
    },
}

# Pretrained embedding — glove-wiki pre formálnejšie novinárske texty
PRETRAINED_MODEL = "glove-wiki-gigaword-100"

MAX_SAMPLES = None  # None = všetky

os.makedirs(OUT_DIR, exist_ok=True)


# ── Text čistenie ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> List[str]:
    """
    Vyčistí text príspevku a vráti zoznam tokenov.
    - lowercase (uncased)
    - odstráni URL, HTML tagy, špeciálne znaky
    - zachová slová dlhšie ako 1 znak
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)       # URL
    text = re.sub(r'<[^>]+>', ' ', text)               # HTML tagy
    text = re.sub(r'[^a-z\s]', ' ', text)              # len písmená
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# ── TF-IDF váhy ───────────────────────────────────────────────────────────────
def compute_tfidf_weights(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """
    Vypočíta IDF váhy pre každé slovo v korpuse.
    IDF = log(N / (df + 1))
    """
    N = len(tokenized_docs)
    df = defaultdict(int)
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[word] += 1
    idf = {word: np.log(N / (count + 1)) for word, count in df.items()}
    return idf


# ── Dokument → vektor ─────────────────────────────────────────────────────────
def doc_to_vector(tokens: List[str], kv, idf_weights: Dict,
                  vector_size: int) -> np.ndarray:
    """
    TF-IDF vážený priemer pretrained embedding vektorov.
    Fallback na čistý priemer ak TF-IDF váhy zlyhajú.
    """
    if not tokens:
        return np.zeros(vector_size)

    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    n_tokens = len(tokens)

    weighted_sum = np.zeros(vector_size)
    total_weight = 0.0

    for word, count in tf.items():
        if word not in kv:
            continue
        tf_val = count / n_tokens
        idf_val = idf_weights.get(word, 0.0)
        weight = tf_val * idf_val
        if weight <= 0:
            continue
        weighted_sum += weight * kv[word]
        total_weight += weight

    if total_weight == 0:
        in_vocab = [w for w in tokens if w in kv]
        if not in_vocab:
            return np.zeros(vector_size)
        return np.mean([kv[w] for w in in_vocab], axis=0)

    return weighted_sum / total_weight


# ── Spracovanie jedného datasetu ──────────────────────────────────────────────
def preprocess_one(name: str, info: dict, kv, vector_size: int):
    """Predspracuje jeden fake news dataset."""

    raw_path = info["raw"]
    out_path = info["out"]

    print(f"\n{'═'*60}")
    print(f"  {name}: {info['desc']}")
    print(f"{'═'*60}")

    if not os.path.exists(raw_path):
        print(f"  CHYBA: Súbor nenájdený: {raw_path}")
        return

    # 1. Načítanie
    print(f"  Načítavam z: {raw_path}")
    df = pd.read_csv(raw_path)

    # Kontrola stĺpcov
    expected_cols = {'text', 'label'}
    if not expected_cols.issubset(set(df.columns)):
        print(f"  CHYBA: Očakávané stĺpce {expected_cols}, nájdené {set(df.columns)}")
        return

    df.dropna(subset=["text", "label"], inplace=True)

    # Ignorovať stĺpec 'topic' — slúži len na simuláciu driftu
    if 'topic' in df.columns:
        topics = df['topic'].unique()
        print(f"  Témy (ignorované): {topics.tolist()}")
        topic_counts = df['topic'].value_counts()
        for t, c in topic_counts.items():
            print(f"    {t}: {c} vzoriek")

    print(f"  Celkový počet riadkov: {len(df)}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES].copy()
        print(f"  Orezané na: {MAX_SAMPLES} vzoriek")

    # Zachovať pôvodné poradie (simulácia streamu)
    y = df["label"].values.astype(int)
    print(f"  Distribúcia tried: {dict(zip(*np.unique(y, return_counts=True)))}")
    minority_pct = min(np.bincount(y)) / len(y) * 100
    print(f"  Minority class: {minority_pct:.1f}%")

    # 2. Tokenizácia
    print("  Tokenizujem texty (uncased)...")
    tokenized = [clean_text(text) for text in df["text"].values]
    avg_len = np.mean([len(t) for t in tokenized])
    print(f"  Priemerná dĺžka príspevku: {avg_len:.1f} tokenov")

    # Pokrytie pretrained modelom
    all_tokens = set(t for doc in tokenized for t in doc)
    in_vocab = sum(1 for t in all_tokens if t in kv)
    print(f"  Pokrytie unikátnych tokenov: {in_vocab}/{len(all_tokens)} "
          f"({in_vocab/len(all_tokens)*100:.1f}%)")

    # 3. TF-IDF váhy
    print("  Počítam TF-IDF váhy...")
    idf_weights = compute_tfidf_weights(tokenized)

    # 4. Dokument → vektor
    print("  Konvertujem na vektory (TF-IDF × pretrained)...")
    X = np.vstack([
        doc_to_vector(tokens, kv, idf_weights, vector_size)
        for tokens in tokenized
    ])
    print(f"  Výsledná matica: {X.shape}")

    zero_rows = np.all(X == 0, axis=1).sum()
    print(f"  Dokumenty bez pokrytia (nulový vektor): {zero_rows} "
          f"({zero_rows/len(X)*100:.1f}%)")

    # 5. MinMax škálovanie
    print("  Škálujem príznaky (MinMax)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Uloženie
    print(f"  Ukladám do: {out_path}")
    data = np.concatenate([X_scaled, y.reshape(-1, 1)], axis=1)
    np.savetxt(out_path, data, delimiter=",", fmt="%.8f")

    print(f"\n  ✓  {name:<20} {X.shape[0]:>7} vzoriek  "
          f"{X.shape[1]:>3} príznakov  "
          f"triedy={sorted(set(y))}  → {out_path}")
    print(f"  Pretrained model: {PRETRAINED_MODEL} ({vector_size}d)")


# ── Hlavná funkcia ─────────────────────────────────────────────────────────────
def preprocess_fakenews():
    try:
        import gensim.downloader as api
    except ImportError:
        print("CHYBA: gensim nie je nainštalovaný. Spusti: pip install gensim")
        return

    # Načítanie pretrained modelu (raz pre oba datasety)
    print(f"\nNačítavam pretrained model: {PRETRAINED_MODEL}")
    print("  (Prvé spustenie stiahne model — môže trvať niekoľko minút)")
    kv = api.load(PRETRAINED_MODEL)
    vector_size = kv.vector_size
    print(f"  Veľkosť slovníka: {len(kv):,} slov, dimenzia: {vector_size}")

    # Spracovanie oboch datasetov
    for name, info in DATASETS.items():
        preprocess_one(name, info, kv, vector_size)

    print(f"\n{'═'*60}")
    print("Hotovo. Pridaj datasety do REAL_DATASETS v runneri:")
    print('  "ElectCovid": "real/real_clean/elect_covid_clean.csv",')
    print('  "FakeNewsComb": "real/real_clean/comb_clean.csv",')
    print(f"{'═'*60}")


if __name__ == "__main__":
    preprocess_fakenews()