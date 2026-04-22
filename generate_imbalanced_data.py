"""
Generate synthetic imbalanced datasets for streaming experiments.

Produces:
    - Binary 90/10 datasets with different drift patterns
    - Multiclass datasets with abrupt, gradual, and reoccurring drift

Output directories:
    - data/synthetic_imbalanced/
    - data/synthetic_multiclass/

Run:
    python generate_imbalanced_data.py
"""

import os
import numpy as np
import pandas as pd

from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data import HyperplaneGenerator
from imblearn.datasets import make_imbalance

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

DATA_DIR_BINARY = "./data/synthetic_imbalanced/"
DATA_DIR_MULTI = "./data/synthetic_multiclass/"
os.makedirs(DATA_DIR_BINARY, exist_ok=True)
os.makedirs(DATA_DIR_MULTI, exist_ok=True)

N_SAMPLES = 50_000
MAJ_RATIO = 0.90

imbalance_config_binary = {
    0: int(N_SAMPLES * MAJ_RATIO),
    1: N_SAMPLES - int(N_SAMPLES * MAJ_RATIO)
}

balanced_config_binary = {
    0: int(N_SAMPLES * 0.5),
    1: N_SAMPLES - int(N_SAMPLES * 0.5)
}


# =============================================================================
# BINARY DATASETY
# =============================================================================

def generate_with_target_counts(stream_like, target_counts, batch_size=50_000,
                                max_batches=200, chunk_size=1000,
                                preserve_order=True, random_state=42):
    """
    Pull samples from stream_like and downsample them so that the global
    class ratio approximately matches target_counts, *while preserving the
    temporal ordering* of the stream.

    Strategy for preserve_order=True:

      1. Identify which class should be the final "desired-majority" and
         which should be the "desired-minority", based on target_counts.
      2. Walk the stream chunk by chunk. In each chunk keep ALL samples of
         the desired-majority class and downsample the desired-minority
         class so that the local ratio matches the target ratio.
      3. This inverts the naive "keep the rare class, downsample the common
         class" approach. For generators like RandomRBFGeneratorDrift which
         produce a roughly 50/50 stream, we enforce a 90/10 (or whatever)
         ratio by aggressively downsampling the class we *want* to be rare,
         not the class the generator happens to produce in smaller numbers.

    This preserves drift (relative ordering of samples is untouched) and
    gives a predictable final ratio regardless of the generator's natural
    class balance.

    Parameters
    ----------
    stream_like   : scikit-multiflow stream exposing next_sample(n)
    target_counts : dict {class_label: int} --- desired per-class counts.
                    The class with the largest count is treated as majority.
    batch_size    : samples per stream batch pull
    max_batches   : hard cap on stream batches
    chunk_size    : granularity of the per-chunk rebalancing
    preserve_order: if False, falls back to legacy global make_imbalance +
                    full permutation (kept only for debugging / non-drift)
    random_state  : RNG seed

    Returns
    -------
    X_final, y_final, total_generated, raw_counts
    """
    need_counts = dict(target_counts)
    n_classes = max(need_counts.keys()) + 1
    total_needed = sum(need_counts.values())
    target_ratio = {c: need / total_needed for c, need in need_counts.items()}

    # Desired-majority = class with the largest target count.
    # Desired-minority = class with the smallest target count.
    # (For multiclass we would generalise this, but binary covers our use.)
    desired_maj = max(target_ratio, key=lambda c: target_ratio[c])
    desired_min = min(target_ratio, key=lambda c: target_ratio[c])
    maj_ratio_target = target_ratio[desired_maj]   # e.g. 0.9
    min_ratio_target = target_ratio[desired_min]   # e.g. 0.1

    # ---- pull raw material ----------------------------------------------
    X_chunks, y_chunks = [], []
    total_generated = 0

    for _ in range(max_batches):
        Xb, yb = stream_like.next_sample(batch_size)
        X_chunks.append(Xb)
        y_chunks.append(yb.astype(int))
        total_generated += len(yb)

        y_all = np.concatenate(y_chunks)
        counts = np.bincount(y_all, minlength=n_classes)

        # Stop once we have at least enough of the desired-majority class
        # (that is the limiting resource under this strategy).
        if counts[desired_maj] >= need_counts[desired_maj]:
            X_all = np.vstack(X_chunks)
            y_all = y_all.astype(int)
            break
    else:
        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks).astype(int)
        counts = np.bincount(y_all, minlength=n_classes)
        print(f"  WARN: reached max_batches={max_batches}; proceeding with "
              f"{total_generated} samples, raw counts={counts.tolist()}")

    # ---- legacy non-drift path ------------------------------------------
    if not preserve_order:
        X_imb, y_imb = make_imbalance(
            X_all, y_all,
            sampling_strategy=need_counts,
            random_state=random_state,
        )
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(y_imb))
        return X_imb[idx], y_imb[idx], total_generated, counts

    # ---- drift-preserving path ------------------------------------------
    rng = np.random.RandomState(random_state)
    X_out, y_out = [], []
    accumulated = {c: 0 for c in need_counts}

    n_total = len(y_all)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        Xc = X_all[start:end]
        yc = y_all[start:end]

        maj_idx = np.where(yc == desired_maj)[0]
        min_idx = np.where(yc == desired_min)[0]

        n_maj_available = len(maj_idx)
        n_min_available = len(min_idx)

        # If the desired-majority class is absent in this chunk we cannot
        # anchor on it --- skip the chunk to avoid collapsing to all-minority.
        if n_maj_available == 0:
            continue

        # Keep all majority samples (up to how many we still need), then
        # compute how many minority samples correspond to the target ratio.
        still_need_maj = need_counts[desired_maj] - accumulated[desired_maj]
        take_maj = min(n_maj_available, still_need_maj)
        if take_maj <= 0:
            break  # majority quota already filled -> stop here

        # target_total_in_chunk such that take_maj / target_total = maj_ratio
        target_total = int(round(take_maj / maj_ratio_target))
        take_min_target = target_total - take_maj
        take_min = min(n_min_available, take_min_target)

        # Select indices inside the chunk.
        if take_maj < n_maj_available:
            sel_maj = rng.choice(maj_idx, size=take_maj, replace=False)
        else:
            sel_maj = maj_idx

        if take_min > 0:
            sel_min = rng.choice(min_idx, size=take_min, replace=False)
            sel = np.sort(np.concatenate([sel_maj, sel_min]))
        else:
            sel = np.sort(sel_maj)

        X_out.append(Xc[sel])
        y_out.append(yc[sel])

        accumulated[desired_maj] += take_maj
        accumulated[desired_min] += take_min

        if accumulated[desired_maj] >= need_counts[desired_maj]:
            break

    if not X_out:
        raise RuntimeError(
            "No usable chunks produced --- check the generator configuration."
        )

    X_final = np.vstack(X_out)
    y_final = np.concatenate(y_out)

    final_counts = np.bincount(y_final, minlength=n_classes)
    final_ratio = final_counts / final_counts.sum() if final_counts.sum() > 0 else final_counts
    print(f"  final_counts={final_counts.tolist()}, "
          f"ratio={dict(zip(range(n_classes), np.round(final_ratio, 3).tolist()))}")

    return X_final, y_final, total_generated, counts

def generate_balanced_datasets():
    """
    Generate 1:1 class-ratio versions of SEA and Agrawal for sanity-checking
    robustness on non-imbalanced streams. Configuration mirrors the imbalanced
    variants so that direct comparisons are meaningful.
    """
    print("Generating balanced datasets...")

    # SEA balanced — abrupt drift
    print("Generating balanced SEA with abrupt drift...")
    stream1 = SEAGenerator(classification_function=0, random_state=42)
    stream2 = SEAGenerator(classification_function=2, random_state=42)

    drift_stream = ConceptDriftStream(
        stream=stream1,
        drift_stream=stream2,
        position=N_SAMPLES // 2,
        width=100,
    )

    # Potlačíme varovanie "overflow encountered in exp" z knižnice skmultiflow
    with np.errstate(over='ignore'):
        X, y, total_generated, counts = generate_with_target_counts(
            drift_stream,
            target_counts=balanced_config_binary,
            batch_size=50_000,
            max_batches=100
        )

    df_sea = pd.DataFrame(X, columns=[f"attr_{i}" for i in range(X.shape[1])])
    df_sea["class"] = y.astype(int)
    df_sea.to_csv(os.path.join(DATA_DIR_BINARY, "sea_balanced.csv"), index=False)

    print(f"Done. (Generated {total_generated} samples to find 50/50 balance)")
    print("Class distribution:")
    print(df_sea["class"].value_counts(normalize=True).sort_index())
    print()

    # Agrawal balanced — abrupt drift
    print("Generating balanced Agrawal with abrupt drift...")
    stream1 = AGRAWALGenerator(classification_function=0, random_state=42)
    stream2 = AGRAWALGenerator(classification_function=1, random_state=42)

    drift_stream = ConceptDriftStream(
        stream=stream1,
        drift_stream=stream2,
        position=N_SAMPLES // 2,
        width=100,
    )

    with np.errstate(over='ignore'):
        X, y, total_generated, counts = generate_with_target_counts(
            drift_stream,
            target_counts=balanced_config_binary,
            batch_size=50_000,
            max_batches=100
        )

    df_agr = pd.DataFrame(X, columns=[f"attr_{i}" for i in range(X.shape[1])])
    df_agr["class"] = y.astype(int)
    df_agr.to_csv(os.path.join(DATA_DIR_BINARY, "agrawal_balanced.csv"), index=False)

    print(f"Done. (Generated {total_generated} samples to find 50/50 balance)")
    print("Class distribution:")
    print(df_agr["class"].value_counts(normalize=True).sort_index())
    print()


def generate_binary_datasets():
    print("Generating binary datasets...")

    # 1) SEA abrupt
    print("Generating heavily imbalanced SEA_Abrupt...")
    stream_sea = SEAGenerator(classification_function=0, random_state=42)

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        stream_sea,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_sea = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_sea["class"] = y_imb
    df_sea.to_csv(os.path.join(DATA_DIR_BINARY, "sea_abrupt_imb9010.csv"), index=False)

    print(f"Done. (total generated: {total_generated}, pool counts: {counts})")
    print(df_sea["class"].value_counts(normalize=True).sort_index())
    print()

    # 2) Agrawal drift
    print("Generating heavily imbalanced Agrawal with drift...")
    stream1 = AGRAWALGenerator(classification_function=0, random_state=42)
    stream2 = AGRAWALGenerator(classification_function=1, random_state=42)

    drift_stream = ConceptDriftStream(
        stream=stream1,
        drift_stream=stream2,
        position=N_SAMPLES // 2,
        width=100
    )

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        drift_stream,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_agr = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_agr["class"] = y_imb
    df_agr.to_csv(os.path.join(DATA_DIR_BINARY, "agrawal_drift_imb9010.csv"), index=False)

    print(f"Done. (total generated: {total_generated}, pool counts: {counts})")
    print(df_agr["class"].value_counts(normalize=True).sort_index())
    print()

    # 3) Hyperplane gradual drift
    print("Generating heavily imbalanced Rotating Hyperplane...")
    hp_stream = HyperplaneGenerator(
        random_state=42,
        n_features=10,
        n_drift_features=5,
        mag_change=0.001,
        noise_percentage=0.05
    )

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        hp_stream,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_hp = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_hp["class"] = y_imb
    df_hp.to_csv(os.path.join(DATA_DIR_BINARY, "hyperplane_gradual_imb9010.csv"), index=False)

    print(f"Done. (total generated: {total_generated}, pool counts: {counts})")
    print(df_hp["class"].value_counts(normalize=True).sort_index())
    print()

# =============================================================================
# MULTICLASS DATASETY
# =============================================================================

def normalize_weights(class_weights):
    w = np.array(class_weights, dtype=float)
    w = w / w.sum()
    return w.tolist()


def make_multiclass_chunk(
    n_samples,
    n_features,
    n_classes,
    class_weights,
    class_sep=1.0,
    flip_y=0.01,
    n_informative=None,
    shift=None,
    scale=None,
    random_state=42,
):
    if n_informative is None:
        n_informative = max(3, min(n_features, n_classes * 2))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=normalize_weights(class_weights),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    if shift is not None:
        shift = np.asarray(shift, dtype=float)
        X = X + shift

    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        X = X * scale

    return X.astype(float), y.astype(int)


def blend_chunks(X_a, y_a, X_b, y_b, width):
    assert len(X_a) == len(X_b) == len(y_a) == len(y_b)
    n = len(y_a)

    if width <= 0:
        return X_b, y_b

    start = max(0, (n // 2) - (width // 2))
    end = min(n, start + width)

    X_out = []
    y_out = []

    for i in range(n):
        if i < start:
            X_out.append(X_a[i])
            y_out.append(y_a[i])
        elif i >= end:
            X_out.append(X_b[i])
            y_out.append(y_b[i])
        else:
            p_b = (i - start) / max(1, (end - start))
            if np.random.rand() < p_b:
                X_out.append(X_b[i])
                y_out.append(y_b[i])
            else:
                X_out.append(X_a[i])
                y_out.append(y_a[i])

    return np.asarray(X_out, dtype=float), np.asarray(y_out, dtype=int)


def scale_features_0_1(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def save_multiclass_dataset(X, y, filename):
    cols = [f"attr_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["class"] = y.astype(int)
    df.to_csv(os.path.join(DATA_DIR_MULTI, filename), index=False)

    print(f"\nSaved: {filename}")
    print("Shape:", df.shape)
    print("Class distribution:")
    print(df["class"].value_counts().sort_index())
    print("Class ratios:")
    print(df["class"].value_counts(normalize=True).sort_index())


def generate_mc_abrupt_3c_70155():
    n1 = N_SAMPLES // 2
    n2 = N_SAMPLES - n1

    X1, y1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=1.2,
        flip_y=0.01,
        shift=np.zeros(12),
        scale=np.ones(12),
        random_state=101,
    )

    X2, y2 = make_multiclass_chunk(
        n_samples=n2,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=0.9,
        flip_y=0.02,
        shift=np.array([0.8, -0.7, 0.5, -0.3, 0.6, -0.5, 0.2, -0.1, 0.4, -0.6, 0.7, -0.2]),
        scale=np.array([1.1, 0.9, 1.0, 1.2, 1.0, 0.8, 1.1, 1.0, 0.9, 1.2, 1.0, 1.1]),
        random_state=102,
    )

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_abrupt_3c_70155.csv")


def generate_mc_gradual_3c_70155():
    n = N_SAMPLES

    X_a, y_a = make_multiclass_chunk(
        n_samples=n,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=1.1,
        flip_y=0.01,
        shift=np.zeros(12),
        scale=np.ones(12),
        random_state=201,
    )

    X_b, y_b = make_multiclass_chunk(
        n_samples=n,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=0.85,
        flip_y=0.03,
        shift=np.array([-0.9, 0.7, -0.4, 0.5, -0.6, 0.3, -0.7, 0.2, -0.5, 0.6, -0.2, 0.4]),
        scale=np.array([0.9, 1.2, 1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.1, 0.8, 1.0, 1.2]),
        random_state=202,
    )

    X, y = blend_chunks(X_a, y_a, X_b, y_b, width=8000)
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_gradual_3c_70155.csv")


def generate_mc_abrupt_4c_601555():
    n1 = N_SAMPLES // 2
    n2 = N_SAMPLES - n1

    X1, y1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=14,
        n_classes=4,
        class_weights=[0.60, 0.15, 0.15, 0.10],
        class_sep=1.15,
        flip_y=0.01,
        shift=np.zeros(14),
        scale=np.ones(14),
        random_state=301,
    )

    X2, y2 = make_multiclass_chunk(
        n_samples=n2,
        n_features=14,
        n_classes=4,
        class_weights=[0.60, 0.15, 0.15, 0.10],
        class_sep=0.80,
        flip_y=0.03,
        shift=np.array([0.7, -0.4, 0.5, -0.6, 0.4, -0.3, 0.6, -0.7, 0.2, -0.5, 0.3, -0.2, 0.8, -0.1]),
        scale=np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 1.0, 0.9]),
        random_state=302,
    )

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_abrupt_4c_601555.csv")


def generate_mc_reoccurring_3c_80155():
    n1 = N_SAMPLES // 3
    n2 = N_SAMPLES // 3
    n3 = N_SAMPLES - n1 - n2

    Xa1, ya1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=1.25,
        flip_y=0.01,
        shift=np.zeros(10),
        scale=np.ones(10),
        random_state=401,
    )

    Xb, yb = make_multiclass_chunk(
        n_samples=n2,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=0.85,
        flip_y=0.03,
        shift=np.array([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.6, -0.5]),
        scale=np.array([1.2, 0.8, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 1.0, 0.9]),
        random_state=402,
    )

    Xa2, ya2 = make_multiclass_chunk(
        n_samples=n3,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=1.20,
        flip_y=0.015,
        shift=np.array([0.15, -0.10, 0.05, -0.08, 0.03, -0.02, 0.01, -0.04, 0.07, -0.03]),
        scale=np.ones(10),
        random_state=403,
    )

    X = np.vstack([Xa1, Xb, Xa2])
    y = np.concatenate([ya1, yb, ya2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_reoccurring_3c_80155.csv")


def generate_multiclass_datasets():
    print("Generating multiclass datasets...")
    np.random.seed(42)

    generate_mc_abrupt_3c_70155()
    generate_mc_gradual_3c_70155()
    generate_mc_abrupt_4c_601555()
    generate_mc_reoccurring_3c_80155()


if __name__ == "__main__":
    generate_balanced_datasets()
    generate_binary_datasets()
    generate_multiclass_datasets()

    print("\nDone.")
    print("Binary datasets saved to:", DATA_DIR_BINARY)
    print("Multiclass datasets saved to:", DATA_DIR_MULTI)
