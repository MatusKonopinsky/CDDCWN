"""
Preprocess Fake News datasets for streaming experiments.

Datasets:
    1. elect_covid.csv - fake-post detection, simulated topic shift
         from "election" to "covid" (concept drift via domain change)
    2. comb.csv - multi-topic version with several topic shifts

Both datasets share the same structure: topic, text, label
    - topic: post topic (ignored - used only to simulate drift)
    - text: post text
    - label: 0 = real, 1 = fake
    - Row order simulates a stream (no timestamp, but pre-ordered)

Pipeline:
    1. Load CSV and ignore the 'topic' column
    2. Clean text (lowercase, remove URLs and special characters)
    3. Load pretrained embedding model (gensim.downloader)
    4. Compute TF-IDF weights over the full corpus
    5. Represent document as TF-IDF weighted mean of pretrained vectors
    6. MinMax scaling + save as clean CSV

Input:  ./data/real/real_text/elect_covid.csv
                ./data/real/real_text/comb.csv
Output: ./data/real/real_clean/elect_covid_clean.csv
                ./data/real/real_clean/comb_clean.csv
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict


# Parameters
DATA_DIR = "./data/real/real_text/"
OUT_DIR = "./data/real/real_clean/"

DATASETS = {
    "elect_covid": {
        "raw": os.path.join(DATA_DIR, "elect_covid.csv"),
        "out": os.path.join(OUT_DIR, "elect_covid_clean.csv"),
        "desc": "Election->COVID concept drift (2 topics)",
    },
    "comb": {
        "raw": os.path.join(DATA_DIR, "comb.csv"),
        "out": os.path.join(OUT_DIR, "comb_clean.csv"),
        "desc": "Multi-topic concept drift (multiple topics)",
    },
}

# Pretrained embedding - glove-wiki for more formal news-like text
PRETRAINED_MODEL = "glove-wiki-gigaword-100"

MAX_SAMPLES = None  # None = all

os.makedirs(OUT_DIR, exist_ok=True)


# Text cleaning
def clean_text(text: str) -> List[str]:
    """
    Clean post text and return token list.
    - lowercase (uncased)
    - remove URLs, HTML tags, and special characters
    - keep words longer than 1 character
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)       # URL
    text = re.sub(r'<[^>]+>', ' ', text)               # HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)              # letters only
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# TF-IDF weights
def compute_tfidf_weights(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """
    Compute IDF weights for each corpus word.
    IDF = log(N / (df + 1))
    """
    N = len(tokenized_docs)
    df = defaultdict(int)
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[word] += 1
    idf = {word: np.log(N / (count + 1)) for word, count in df.items()}
    return idf


# Document -> vector
def doc_to_vector(tokens: List[str], kv, idf_weights: Dict,
                  vector_size: int) -> np.ndarray:
    """
    TF-IDF weighted mean of pretrained embedding vectors.
    Fall back to plain mean if TF-IDF weighting fails.
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


# Processing one dataset
def preprocess_one(name: str, info: dict, kv, vector_size: int):
    """Preprocess one fake-news dataset."""

    raw_path = info["raw"]
    out_path = info["out"]

    print(f"\n{'═'*60}")
    print(f"  {name}: {info['desc']}")
    print(f"{'═'*60}")

    if not os.path.exists(raw_path):
        print(f"  ERROR: File not found: {raw_path}")
        return

    # 1. Loading
    print(f"  Loading from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Column validation
    expected_cols = {'text', 'label'}
    if not expected_cols.issubset(set(df.columns)):
        print(f"  ERROR: Expected columns {expected_cols}, found {set(df.columns)}")
        return

    df.dropna(subset=["text", "label"], inplace=True)

    # Ignore 'topic' column - used only for drift simulation
    if 'topic' in df.columns:
        topics = df['topic'].unique()
        print(f"  Topics (ignored): {topics.tolist()}")
        topic_counts = df['topic'].value_counts()
        for t, c in topic_counts.items():
            print(f"    {t}: {c} samples")

    print(f"  Total rows: {len(df)}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES].copy()
        print(f"  Truncated to: {MAX_SAMPLES} samples")

    # Preserve original order (stream simulation)
    y = df["label"].values.astype(int)
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    minority_pct = min(np.bincount(y)) / len(y) * 100
    print(f"  Minority class: {minority_pct:.1f}%")

    # 2. Tokenization
    print("  Tokenizing texts (uncased)...")
    tokenized = [clean_text(text) for text in df["text"].values]
    avg_len = np.mean([len(t) for t in tokenized])
    print(f"  Mean post length: {avg_len:.1f} tokens")

    # Pretrained model coverage
    all_tokens = set(t for doc in tokenized for t in doc)
    in_vocab = sum(1 for t in all_tokens if t in kv)
    print(f"  Unique token coverage: {in_vocab}/{len(all_tokens)} "
          f"({in_vocab/len(all_tokens)*100:.1f}%)")

    # 3. TF-IDF weights
    print("  Computing TF-IDF weights...")
    idf_weights = compute_tfidf_weights(tokenized)

    # 4. Document -> vector
    print("  Converting to vectors (TF-IDF x pretrained)...")
    X = np.vstack([
        doc_to_vector(tokens, kv, idf_weights, vector_size)
        for tokens in tokenized
    ])
    print(f"  Result matrix: {X.shape}")

    zero_rows = np.all(X == 0, axis=1).sum()
    print(f"  Documents with no coverage (zero vector): {zero_rows} "
          f"({zero_rows/len(X)*100:.1f}%)")

    # 5. MinMax scaling
    print("  Scaling features (MinMax)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Save output
    print(f"  Saving to: {out_path}")
    data = np.concatenate([X_scaled, y.reshape(-1, 1)], axis=1)
    np.savetxt(out_path, data, delimiter=",", fmt="%.8f")

    print(f"\n    {name:<20} {X.shape[0]:>7} samples  "
          f"{X.shape[1]:>3} features  "
          f"classes={sorted(set(y))}  -> {out_path}")
    print(f"  Pretrained model: {PRETRAINED_MODEL} ({vector_size}d)")


# Main function
def preprocess_fakenews():
    try:
        import gensim.downloader as api
    except ImportError:
        print("ERROR: gensim is not installed. Run: pip install gensim")
        return

    # Load pretrained model (once for both datasets)
    print(f"\nLoading pretrained model: {PRETRAINED_MODEL}")
    print("  (First run will download the model - this may take a few minutes)")
    kv = api.load(PRETRAINED_MODEL)
    vector_size = kv.vector_size
    print(f"  Vocabulary size: {len(kv):,} words, dimension: {vector_size}")

    # Process both datasets
    for name, info in DATASETS.items():
        preprocess_one(name, info, kv, vector_size)

    print(f"\n{'═'*60}")
    print("Done. Add datasets to REAL_DATASETS in the runner:")
    print('  "ElectCovid": "real/real_clean/elect_covid_clean.csv",')
    print('  "FakeNewsComb": "real/real_clean/comb_clean.csv",')
    print(f"{'═'*60}")


if __name__ == "__main__":
    preprocess_fakenews()