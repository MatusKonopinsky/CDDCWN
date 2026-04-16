"""
Preprocess Jigsaw Toxic Comment dataset for streaming experiments.

Pipeline:
    1. Load and clean text (lowercase, remove special characters)
    2. Load PRETRAINED embedding model (gensim.downloader)
    3. Compute TF-IDF weights for each word in the corpus
    4. Represent each comment as TF-IDF weighted mean of pretrained vectors
    5. Save as clean CSV (MinMax scaled, no header, last column = label)

Pretrained models:
    - "glove-twitter-100"       - 1.2M words, 100d, Twitter (ideal for informal comments)
    - "glove-wiki-gigaword-100" - 400k words, 100d, Wikipedia + Gigaword (smaller, more formal)
    - "word2vec-google-news-300"- 3M words, 300d, Google News (largest coverage, 1.7GB)

For Jigsaw (toxic comments = informal internet text), glove-twitter is typically
the best fit - it covers slang, abbreviations, and profanity better than Wikipedia-based models.

Input:  ./data/real/real_text/train.csv
Output: ./data/real/real_clean/jigsaw_clean.csv
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict

# Parameters
RAW_PATH    = "./data/real/real_text/train.csv"
OUT_PATH    = "./data/real/real_clean/jigsaw_clean.csv"
TARGET_COL  = "toxic"
MAX_SAMPLES = None            # None = all (~160k), e.g. 100_000 for faster runs

# Pretrained embedding model (downloaded automatically via gensim.downloader)
# Options: "glove-twitter-100", "glove-twitter-200",
#          "glove-wiki-gigaword-100", "glove-wiki-gigaword-300",
#          "word2vec-google-news-300"
PRETRAINED_MODEL = "glove-twitter-100"

RANDOM_STATE = 42

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# Text cleaning
def clean_text(text: str) -> List[str]:
    """
        Clean comment text and return token list.
        - lowercase (uncased) - casing usually does not help toxic detection,
            pretrained GloVe Twitter is also lowercase
        - remove URLs, numbers, and special characters
        - keep words longer than 1 character
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)       # URL
    text = re.sub(r'[^a-z\s]', ' ', text)             # letters only
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# TF-IDF weights
def compute_tfidf_weights(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """
    Compute TF-IDF weights for each word in the corpus.
    Returns dict {word: idf_weight}; TF is computed per document during embedding.
    Uses IDF = log(N / df), where df is number of documents containing the word.
    """
    N = len(tokenized_docs)
    df = defaultdict(int)
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[word] += 1

    idf = {word: np.log(N / (count + 1)) for word, count in df.items()}
    return idf


# Document -> vector (pretrained)
def doc_to_vector_pretrained(tokens: List[str], kv, idf_weights: Dict,
                             vector_size: int) -> np.ndarray:
    """
    Convert token list to document vector using TF-IDF weighted
    average of pretrained embedding vectors.

    Parameters
    ----------
    tokens      : token list (lowercase)
    kv          : gensim KeyedVectors (pretrained embeddings)
    idf_weights : dict {word: IDF weight}
    vector_size : embedding dimension

    Returns
    -------
    numpy vector of length vector_size
    """
    if not tokens:
        return np.zeros(vector_size)

    # Compute TF
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
        # Fallback: plain mean for in-vocabulary words
        in_vocab = [w for w in tokens if w in kv]
        if not in_vocab:
            return np.zeros(vector_size)
        return np.mean([kv[w] for w in in_vocab], axis=0)

    return weighted_sum / total_weight


# Main function
def preprocess_jigsaw():
    try:
        import gensim.downloader as api
    except ImportError:
        print("ERROR: gensim is not installed. Run: pip install gensim")
        return

    # 1. Loading
    print(f"\nLoading Jigsaw dataset from: {RAW_PATH}")
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: File not found: {RAW_PATH}")
        return

    df = pd.read_csv(RAW_PATH)
    df.dropna(subset=["comment_text", TARGET_COL], inplace=True)
    print(f"  Loaded rows: {len(df)}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES].copy()
        print(f"  Truncated to: {MAX_SAMPLES} samples")

    y = df[TARGET_COL].values.astype(int)
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Minority (toxic=1): {y.mean()*100:.1f}%")

    # 2. Tokenization
    print("\nTokenizing texts (uncased)...")
    tokenized = [clean_text(text) for text in df["comment_text"].values]
    avg_len = np.mean([len(t) for t in tokenized])
    print(f"  Mean comment length: {avg_len:.1f} tokens")

    # 3. Load pretrained model
    print(f"\nLoading pretrained model: {PRETRAINED_MODEL}")
    print("  (First run will download the model - this may take a few minutes)")
    kv = api.load(PRETRAINED_MODEL)
    vector_size = kv.vector_size
    vocab_size = len(kv)
    print(f"  Vocabulary size: {vocab_size:,} words")
    print(f"  Vector dimension: {vector_size}")

    # Corpus coverage by pretrained model
    all_tokens = set(t for doc in tokenized for t in doc)
    in_vocab = sum(1 for t in all_tokens if t in kv)
    print(f"  Unique token coverage: {in_vocab}/{len(all_tokens)} "
          f"({in_vocab/len(all_tokens)*100:.1f}%)")

    # 4. TF-IDF weights
    print("\nComputing TF-IDF weights...")
    idf_weights = compute_tfidf_weights(tokenized)
    print(f"  Words with IDF weights: {len(idf_weights):,}")

    # 5. Document -> vector
    print("\nConverting comments to vectors (TF-IDF weighted mean of pretrained embeddings)...")
    X = np.vstack([
        doc_to_vector_pretrained(tokens, kv, idf_weights, vector_size)
        for tokens in tokenized
    ])
    print(f"  Result matrix: {X.shape}")

    # Coverage check
    zero_rows = np.all(X == 0, axis=1).sum()
    print(f"  Documents with no coverage (zero vector): {zero_rows} "
          f"({zero_rows/len(X)*100:.1f}%)")

    # 6. MinMax scaling
    print("\nScaling features (MinMax)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. Save output
    print(f"\nSaving to: {OUT_PATH}")
    data = np.concatenate([X_scaled, y.reshape(-1, 1)], axis=1)
    np.savetxt(OUT_PATH, data, delimiter=",", fmt="%.8f")

    print(f"\n{'─'*60}")
    print(f"    jigsaw               {X.shape[0]:>7} samples  "
          f"{X.shape[1]:>3} features  "
          f"classes={sorted(set(y.astype(int)))}  -> {OUT_PATH}")
    print(f"  Pretrained model: {PRETRAINED_MODEL} ({vector_size}d)")
    print(f"{'─'*60}")
    print("\nDone. Add 'Jigsaw' to REAL_DATASETS in the runner:")
    print('  "Jigsaw": "real/real_clean/jigsaw_clean.csv"')


if __name__ == "__main__":
    preprocess_jigsaw()