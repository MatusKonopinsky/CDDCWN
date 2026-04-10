"""
preprocess_jigsaw.py
────────────────────
Predspracuje Jigsaw Toxic Comment dataset pre streamové experimenty.

Pipeline:
  1. Načítanie a čistenie textu (lowercase, odstránenie špeciálnych znakov)
  2. Načítanie PRETRAINED embedding modelu (gensim.downloader)
  3. Výpočet TF-IDF váh pre každé slovo v korpuse
  4. Reprezentácia každého komentára = TF-IDF vážený priemer pretrained vektorov
  5. Uloženie ako clean CSV (MinMax škálované, bez hlavičky, posledný stĺpec = label)

Pretrained modely:
  - "glove-twitter-100"       — 1.2M slov, 100d, Twitter (ideálny pre neformálne komentáre)
  - "glove-wiki-gigaword-100" — 400k slov, 100d, Wikipedia + Gigaword (menší, formálnejší)
  - "word2vec-google-news-300"— 3M slov, 300d, Google News (najväčšie pokrytie, 1.7GB)

Pre Jigsaw (toxic komentáre = neformálny internetový text) je glove-twitter
najvhodnejší — pokrýva slang, skratky a vulgarizmy lepšie ako Wikipedia-based modely.

Vstup:  ./data/real/real_text/train.csv
Výstup: ./data/real/real_clean/jigsaw_clean.csv

Spustenie:
    python preprocess_jigsaw.py
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict

# ── Parametre ─────────────────────────────────────────────────────────────────
RAW_PATH    = "./data/real/real_text/train.csv"
OUT_PATH    = "./data/real/real_clean/jigsaw_clean.csv"
TARGET_COL  = "toxic"
MAX_SAMPLES = None            # None = všetky (~160k), napr. 100_000 pre rýchlejší beh

# Pretrained embedding model (stiahne sa automaticky cez gensim.downloader)
# Možnosti: "glove-twitter-100", "glove-twitter-200",
#           "glove-wiki-gigaword-100", "glove-wiki-gigaword-300",
#           "word2vec-google-news-300"
PRETRAINED_MODEL = "glove-twitter-100"

RANDOM_STATE = 42

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# ── Text čistenie ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> List[str]:
    """
    Vyčistí komentár a vráti zoznam tokenov.
    - lowercase (uncased) — pre toxic detection case nepomáha,
      pretrained GloVe Twitter je tiež lowercase
    - odstráni URL, čísla, špeciálne znaky
    - zachová slová dlhšie ako 1 znak
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)       # URL
    text = re.sub(r'[^a-z\s]', ' ', text)             # len písmená
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# ── TF-IDF váhy ───────────────────────────────────────────────────────────────
def compute_tfidf_weights(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """
    Vypočíta TF-IDF váhy pre každé slovo v korpuse.
    Vracia slovník {slovo: idf_váha} — TF sa počíta per-dokument pri embedovaní.
    Používame IDF = log(N / df) kde df = počet dokumentov obsahujúcich slovo.
    """
    N = len(tokenized_docs)
    df = defaultdict(int)
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[word] += 1

    idf = {word: np.log(N / (count + 1)) for word, count in df.items()}
    return idf


# ── Dokument → vektor (pretrained) ────────────────────────────────────────────
def doc_to_vector_pretrained(tokens: List[str], kv, idf_weights: Dict,
                             vector_size: int) -> np.ndarray:
    """
    Konvertuje zoznam tokenov na dokument vektor pomocou TF-IDF váženého
    priemeru pretrained embedding vektorov.

    Parametre
    ---------
    tokens      : zoznam tokenov (lowercase)
    kv          : gensim KeyedVectors (pretrained embeddings)
    idf_weights : slovník {slovo: IDF váha}
    vector_size : dimenzia vektorov

    Vracia
    ------
    numpy vektor dĺžky vector_size
    """
    if not tokens:
        return np.zeros(vector_size)

    # Počítaj TF
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    n_tokens = len(tokens)

    weighted_sum = np.zeros(vector_size)
    total_weight = 0.0

    for word, count in tf.items():
        if word not in kv:
            continue
        tf_val  = count / n_tokens
        idf_val = idf_weights.get(word, 0.0)
        weight  = tf_val * idf_val
        if weight <= 0:
            continue
        weighted_sum += weight * kv[word]
        total_weight += weight

    if total_weight == 0:
        # Fallback: čistý priemer pre slová v slovníku
        in_vocab = [w for w in tokens if w in kv]
        if not in_vocab:
            return np.zeros(vector_size)
        return np.mean([kv[w] for w in in_vocab], axis=0)

    return weighted_sum / total_weight


# ── Hlavná funkcia ─────────────────────────────────────────────────────────────
def preprocess_jigsaw():
    try:
        import gensim.downloader as api
    except ImportError:
        print("CHYBA: gensim nie je nainštalovaný. Spusti: pip install gensim")
        return

    # 1. Načítanie
    print(f"\nNačítavam Jigsaw dataset z: {RAW_PATH}")
    if not os.path.exists(RAW_PATH):
        print(f"CHYBA: Súbor nenájdený: {RAW_PATH}")
        return

    df = pd.read_csv(RAW_PATH)
    df.dropna(subset=["comment_text", TARGET_COL], inplace=True)
    print(f"  Načítaných riadkov: {len(df)}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES].copy()
        print(f"  Orezané na: {MAX_SAMPLES} vzoriek")

    y = df[TARGET_COL].values.astype(int)
    print(f"  Distribúcia tried: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Minority (toxic=1): {y.mean()*100:.1f}%")

    # 2. Tokenizácia
    print("\nTokenizujem texty (uncased)...")
    tokenized = [clean_text(text) for text in df["comment_text"].values]
    avg_len = np.mean([len(t) for t in tokenized])
    print(f"  Priemerná dĺžka komentára: {avg_len:.1f} tokenov")

    # 3. Načítanie pretrained modelu
    print(f"\nNačítavam pretrained model: {PRETRAINED_MODEL}")
    print("  (Prvé spustenie stiahne model — môže trvať niekoľko minút)")
    kv = api.load(PRETRAINED_MODEL)
    vector_size = kv.vector_size
    vocab_size = len(kv)
    print(f"  Veľkosť slovníka: {vocab_size:,} slov")
    print(f"  Dimenzia vektorov: {vector_size}")

    # Pokrytie korpusu pretrained modelom
    all_tokens = set(t for doc in tokenized for t in doc)
    in_vocab = sum(1 for t in all_tokens if t in kv)
    print(f"  Pokrytie unikátnych tokenov: {in_vocab}/{len(all_tokens)} "
          f"({in_vocab/len(all_tokens)*100:.1f}%)")

    # 4. TF-IDF váhy
    print("\nPočítam TF-IDF váhy...")
    idf_weights = compute_tfidf_weights(tokenized)
    print(f"  Počet slov s IDF váhou: {len(idf_weights):,}")

    # 5. Dokument → vektor
    print("\nKonvertujem komentáre na vektory (TF-IDF vážený priemer pretrained embeddings)...")
    X = np.vstack([
        doc_to_vector_pretrained(tokens, kv, idf_weights, vector_size)
        for tokens in tokenized
    ])
    print(f"  Výsledná matica: {X.shape}")

    # Kontrola pokrytia
    zero_rows = np.all(X == 0, axis=1).sum()
    print(f"  Dokumenty bez pokrytia (nulový vektor): {zero_rows} "
          f"({zero_rows/len(X)*100:.1f}%)")

    # 6. MinMax škálovanie
    print("\nŠkálujem príznaky (MinMax)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. Uloženie
    print(f"\nUkladám do: {OUT_PATH}")
    data = np.concatenate([X_scaled, y.reshape(-1, 1)], axis=1)
    np.savetxt(OUT_PATH, data, delimiter=",", fmt="%.8f")

    print(f"\n{'─'*60}")
    print(f"  ✓  jigsaw               {X.shape[0]:>7} vzoriek  "
          f"{X.shape[1]:>3} príznakov  "
          f"triedy={sorted(set(y.astype(int)))}  → {OUT_PATH}")
    print(f"  Pretrained model: {PRETRAINED_MODEL} ({vector_size}d)")
    print(f"{'─'*60}")
    print("\nHotovo. Pridaj 'Jigsaw' do REAL_DATASETS v runneri:")
    print('  "Jigsaw": "real/real_clean/jigsaw_clean.csv"')


if __name__ == "__main__":
    preprocess_jigsaw()