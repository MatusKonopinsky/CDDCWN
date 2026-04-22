"""
    Implementation of a drift-aware online ensemble classifier IDDCW (Improved DDCW) The file contains:
    1. SimplePageHinkley: a lightweight concept-drift detector on the error stream.
    2. IDDCW: an adaptive ensemble with expert weighting,
       minority replay/augmentation, and majority-recall regulation.

Main features:
    - soft voting with per-class expert weights,
    - adaptive expert boosting/penalization based on imbalance ratio,
    - replay and optional minority-class augmentation,
    - post-drift cooldown logic and class-buffer handling,
    - online sample-by-sample learning (partial_fit).

Note:
    The class is designed for stream-learning scenarios where data
    distribution changes over time while robustness to class imbalance matters.
"""

import copy as cp
import time
from collections import deque, Counter, defaultdict

import numpy as np
from sklearn.exceptions import NotFittedError

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import (
    HoeffdingTreeClassifier,
    HoeffdingAdaptiveTreeClassifier,
    ExtremelyFastDecisionTreeClassifier,
)
from skmultiflow.neural_networks import PerceptronMask


class SimplePageHinkley:
    """
    Page-Hinkley drift detector on the error stream.
    value = 1.0 -> error, value = 0.0 -> correct prediction

    Parameters
    ----------
    delta     : sensitivity (smaller means more sensitive to subtle changes)
    threshold : detection threshold (larger means fewer false positives)
    alpha     : forgetting factor for estimated error rate mean
    min_detection_interval : minimum number of samples between detections.
                             After a detection, the detector is "deaf" for
                             this many samples, which suppresses cascaded
                             false positives on continuously drifting datasets
                             (e.g., RBF). Replaces the need for manual reset().
    """
    def __init__(self, delta=0.005, threshold=100.0, alpha=0.999,
                 min_detection_interval=2000):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.min_detection_interval = int(min_detection_interval)
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.t = 0
        self._samples_since_last_detection = self.min_detection_interval  # start active

    def update(self, value):
        self.t += 1
        self._samples_since_last_detection += 1

        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value
        self.cum_sum += value - self.mean - self.delta
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)

        # Detect only if min_detection_interval has passed since last detection
        if self._samples_since_last_detection < self.min_detection_interval:
            return False

        if (self.cum_sum - self.min_cum_sum) > self.threshold:
            # Reset cumulative sum and enter waiting period without external reset()
            self.cum_sum = 0.0
            self.min_cum_sum = 0.0
            self._samples_since_last_detection = 0
            return True

        return False


class IDDCW(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    class WeightedExpert:
        def __init__(self, estimator, weight, num_classes):
            self.estimator = estimator
            self.model_type = type(estimator).__name__
            self.weight_class = np.full(num_classes, weight, dtype=float)
            self.lifetime = 0
            self.warmup_remaining = 0

    def __init__(
        self,
        min_estimators=5,
        max_estimators=20,
        base_estimators=None,
        period=600,
        alpha=0.002,
        beta=1.5,
        theta=0.02,
        enable_diversity=False,
        rwa_strength=1.0,
        use_lifetime_trend=True,
        warmup_windows=2,

        # new history parameters
        history_buffer_size=600,
        class_buffer_size=300,

        # replay / augment
        replay_mode="replay",          # "off" | "replay" | "augment"
        replay_k=3,
        augmentation_mode="none",      # "none" | "noise"
        augmentation_strength=0.02,
        imbalance_aware_augmentation=True,

        # replay regulation by majority recall
        majority_recall_critical=0.55,   # below this threshold replay is dampened
        majority_recall_shutoff=0.40,    # below this value replay is fully disabled
        min_replay_support=10,           # min samples in class_buffer for replay

        # drift-aware logic
        enable_drift_detector=False,
        drift_delta=0.005,
        drift_threshold=100.0,
        drift_min_detection_interval=2000,
        post_drift_cooldown=300,
        post_drift_replay_boost=1,
        post_drift_aug_reduction=0.5,
        reset_majority_history_on_drift=True,
        keep_class_buffers_on_drift=True,

        random_state=None,
    ):
        super().__init__()

        if base_estimators is None:
            base_estimators = [
                NaiveBayes(),
                HoeffdingTreeClassifier(),
                HoeffdingAdaptiveTreeClassifier(),
                ExtremelyFastDecisionTreeClassifier(),
                PerceptronMask(),
            ]

        self.min_estimators = int(min_estimators)
        self.max_estimators = int(max_estimators)
        self.base_estimators = base_estimators

        self.period = int(period)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)
        self.enable_diversity = bool(enable_diversity)
        self.rwa_strength = float(rwa_strength)
        self.use_lifetime_trend = bool(use_lifetime_trend)
        self.warmup_windows = int(warmup_windows)

        self.history_buffer_size = int(history_buffer_size)
        self.class_buffer_size = int(class_buffer_size)

        self.replay_mode = str(replay_mode)
        self.replay_k = int(replay_k)
        self.augmentation_mode = str(augmentation_mode)
        self.augmentation_strength = float(augmentation_strength)
        self.imbalance_aware_augmentation = bool(imbalance_aware_augmentation)

        # parameters
        self.majority_recall_critical = float(majority_recall_critical)
        self.majority_recall_shutoff = float(majority_recall_shutoff)
        self.min_replay_support = int(min_replay_support)

        self.enable_drift_detector = bool(enable_drift_detector)
        self.drift_delta = float(drift_delta)
        self.drift_threshold = float(drift_threshold)
        self.drift_min_detection_interval = int(drift_min_detection_interval)
        self.post_drift_cooldown = int(post_drift_cooldown)
        self.post_drift_replay_boost = int(post_drift_replay_boost)
        self.post_drift_aug_reduction = float(post_drift_aug_reduction)
        self.reset_majority_history_on_drift = bool(reset_majority_history_on_drift)
        self.keep_class_buffers_on_drift = bool(keep_class_buffers_on_drift)

        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        self.num_classes = 2
        self._classes = None
        self.experts = []
        self.update_times = deque(maxlen=2000)

        self._y_window = deque(maxlen=self.period)
        self._history_buffer = deque(maxlen=self.history_buffer_size)

        self._warmup_samples = None
        self._seen_samples = 0

        # for multiclass: one buffer per class
        self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))

        self._post_drift_remaining = 0
        self._drift_points = []
        self._drift_detector = None

        # sliding window for majority-recall estimation
        self._recent_true = deque(maxlen=self.period)
        self._recent_preds = deque(maxlen=self.period)

        # Welford running std
        self._welford_n = 0
        self._welford_mean = None
        self._welford_M2 = None

        self.reset()

    # =============================================================================
    # SAFE WRAPPERS
    # =============================================================================

    def _safe_predict(self, estimator, X):
        try:
            return estimator.predict(X)
        except (NotFittedError, AttributeError, ValueError):
            return None
        except Exception:
            return None

    def _safe_predict_proba(self, estimator, X):
        try:
            if hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
                if p is not None and len(p.shape) == 2:
                    return p
        except Exception:
            pass
        return None

    # =============================================================================
    # PARTIAL FIT
    # =============================================================================

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if classes is not None and self._classes is None:
            self._classes = list(classes)
            self.num_classes = max(self.num_classes, len(self._classes))

        if X is None or y is None or len(y) == 0:
            return self

        for i in range(len(y)):
            Xi = X[i:i + 1]
            yi = y[i:i + 1]
            self.fit_single_sample(Xi, yi, classes=self._classes, sample_weight=sample_weight)

        return self

    # =============================================================================
    # SOFT VOTING
    # =============================================================================

    def predict(self, X):
        """
        Soft voting: instead of hard votes, use each expert's predict_proba
        weighted by per-class weights. This mitigates extreme decisions.
        """
        if not self.experts:
            return np.zeros(X.shape[0], dtype=int)

        N = X.shape[0]
        pred_agg = np.zeros((N, self.num_classes), dtype=float)

        for exp in self.experts:
            wc = exp.weight_class
            if len(wc) < self.num_classes:
                wc = np.pad(wc, (0, self.num_classes - len(wc)),
                            "constant", constant_values=1.0)

            p = self._safe_predict_proba(exp.estimator, X)
            if p is None:
                # Fall back to hard vote when estimator lacks predict_proba
                y_hat = self._safe_predict(exp.estimator, X)
                if y_hat is None:
                    continue
                y_hat = np.clip(np.asarray(y_hat, dtype=int), 0, self.num_classes - 1)
                p = np.zeros((N, self.num_classes), dtype=float)
                p[np.arange(N), y_hat] = 1.0

            if p.shape[1] < self.num_classes:
                p = np.pad(p, ((0, 0), (0, self.num_classes - p.shape[1])),
                           "constant", constant_values=0.0)
            elif p.shape[1] > self.num_classes:
                p = p[:, :self.num_classes]

            pred_agg += p * wc[:self.num_classes]

        return np.argmax(pred_agg, axis=1)

    def predict_proba(self, X):
        N = X.shape[0]
        out = np.zeros((N, self.num_classes), dtype=float)

        for exp in self.experts:
            wc = exp.weight_class
            if len(wc) < self.num_classes:
                wc = np.pad(
                    wc,
                    (0, self.num_classes - len(wc)),
                    "constant",
                    constant_values=1.0
                )

            p = self._safe_predict_proba(exp.estimator, X)
            if p is None:
                pred = self._safe_predict(exp.estimator, X)
                if pred is None:
                    continue
                pred = np.clip(np.asarray(pred, dtype=int), 0, self.num_classes - 1)
                p = np.zeros((N, self.num_classes), dtype=float)
                p[np.arange(N), pred] = 1.0

            if p.shape[1] < self.num_classes:
                p = np.pad(
                    p,
                    ((0, 0), (0, self.num_classes - p.shape[1])),
                    "constant",
                    constant_values=0.0
                )
            elif p.shape[1] > self.num_classes:
                p = p[:, :self.num_classes]

            out += p * wc[:self.num_classes]

        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        out /= row_sums
        return out

    # =============================================================================
    # HELPERS
    # =============================================================================

    def train_model(self, model, X, y, classes, sample_weight=None):
        if y is None or len(y) == 0:
            return model
        try:
            try:
                model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
                return model
            except TypeError:
                model.partial_fit(X, y, classes=classes)
                return model
        except Exception:
            try:
                if len(np.unique(y)) < 2:
                    return model
                model.fit(X, y)
            except Exception:
                return model
            return model

    def _window_counts(self):
        if hasattr(self, '_cached_counts_step') and self._cached_counts_step == self._seen_samples:
            return self._cached_counts

        if len(self._y_window) == 0:
            counts = np.zeros(self.num_classes, dtype=int)
        else:
            counts = np.bincount(np.asarray(self._y_window, dtype=int), minlength=self.num_classes)

        self._cached_counts = counts
        self._cached_counts_step = self._seen_samples
        return counts

    def _get_majority_and_minorities(self):
        counts = self._window_counts()
        present = np.where(counts > 0)[0]
        if len(present) == 0:
            return None, []

        majority = int(present[np.argmax(counts[present])])
        minorities = [int(c) for c in present if int(c) != majority]
        return majority, minorities

    def _current_imbalance_ratio(self):
        counts = self._window_counts()
        present = counts[counts > 0]
        if len(present) < 2:
            return 1.0
        return float(np.max(present) / max(1, np.min(present)))

    # Welford running std (replaces periodic np.vstack)
    def _welford_update(self, x_row):
        """Online update of running mean and variance (Welford algorithm)."""
        x = x_row.flatten().astype(float)
        if self._welford_mean is None:
            self._welford_n = 1
            self._welford_mean = x.copy()
            self._welford_M2 = np.zeros_like(x)
        else:
            self._welford_n += 1
            delta = x - self._welford_mean
            self._welford_mean += delta / self._welford_n
            delta2 = x - self._welford_mean
            self._welford_M2 += delta * delta2

    def _local_feature_std(self):
        """
        Return running std from the Welford algorithm.
        Cache interval is 500 steps; std is recomputed only when stale.
        """
        if hasattr(self, '_cached_std_step') and self._seen_samples - self._cached_std_step < 500:
            return self._cached_std

        if self._welford_n < 2:
            return None

        variance = self._welford_M2 / max(1, self._welford_n - 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < 1e-8, 1e-8, std)

        self._cached_std = std
        self._cached_std_step = self._seen_samples
        return std

    # Estimate majority recall from sliding window
    def _estimate_majority_recall(self):
        """
        Fast estimate of majority recall from the latest period samples.
        Used to regulate replay intensity.
        Returns None if there is not enough data yet (< 100 samples).
        """
        if len(self._recent_true) < 100:
            return None
        majority, _ = self._get_majority_and_minorities()
        if majority is None:
            return None

        true_arr = np.array(self._recent_true)
        pred_arr = np.array(self._recent_preds)
        maj_mask = (true_arr == majority)
        n_maj = maj_mask.sum()
        if n_maj == 0:
            return None

        return float((pred_arr[maj_mask] == majority).sum() / n_maj)

    # Adaptive replay_k with majority recall feedback
    def _adaptive_majority_thresholds(self):
        """
        Majority-recall regulation thresholds adapted by imbalance ratio.

        On mildly imbalanced data (ratio < 3, e.g. Airlines 55/45),
        thresholds are strict and majority recall should not drop below 0.55/0.40.

        On extremely imbalanced data (ratio > 8, e.g. RBF 90/10),
        thresholds are much looser; majority recall 0.25 can still be acceptable
        because minority protection is more important on such data.

        Linear interpolation between both regimes in the 3-8 ratio range.

        Returns a (critical, shutoff) tuple.
        """
        ratio = self._current_imbalance_ratio()

        # Regime definitions:
        # Mild imbalance (ratio <= 3): strict majority protection
        crit_low, shut_low = self.majority_recall_critical, self.majority_recall_shutoff
        # Extreme imbalance (ratio >= 8): loose majority protection
        crit_high, shut_high = 0.25, 0.15

        if ratio <= 3.0:
            return crit_low, shut_low
        elif ratio >= 8.0:
            return crit_high, shut_high
        else:
            # Linear interpolation
            t = (ratio - 3.0) / 5.0  # 0.0 at ratio=3, 1.0 at ratio=8
            crit = crit_low + t * (crit_high - crit_low)
            shut = shut_low + t * (shut_high - shut_low)
            return crit, shut

    def _effective_replay_k(self):
        """
          Replay intensity regulated by three factors:
          1. Imbalance ratio - logarithmic cap (ratio 2->k=1, 10->k=3, 100->k=6)
          2. Majority recall feedback - damp/disable when majority suffers,
              with adaptive thresholds based on imbalance ratio
          3. Post-drift boost
        """
        ratio = self._current_imbalance_ratio()

        # Logarithmic cap: replay proportional to imbalance severity
        max_k = max(1, int(np.log2(max(2.0, ratio))))
        k = min(self.replay_k, max_k)

        # Adaptive adjustment by ratio
        if ratio < 3.0:
            k = max(0, k - 1)
        elif ratio > 15.0:
            k = k + 1

        # Majority recall feedback with adaptive thresholds
        maj_recall = self._estimate_majority_recall()
        if maj_recall is not None:
            crit, shut = self._adaptive_majority_thresholds()
            if maj_recall < shut:
                k = 0
            elif maj_recall < crit:
                k = max(0, k // 2)

        # Post-drift boost (if majority is not threatened)
        if self._post_drift_remaining > 0 and k > 0:
            k += self.post_drift_replay_boost

        return max(0, int(k))

    # Augmentation with a lower threshold
    def _effective_aug_strength(self):
        strength = self.augmentation_strength

        if self.imbalance_aware_augmentation:
            ratio = self._current_imbalance_ratio()
            # Disable augmentation only at low imbalance (ratio < 2)
            if ratio < 2.0:
                return 0.0
            strength *= min(3.0, 1.0 + 0.25 * max(0.0, ratio - 2.0))

        # If majority recall is threatened, reduce augmentation
        maj_recall = self._estimate_majority_recall()
        if maj_recall is not None:
            crit, _ = self._adaptive_majority_thresholds()
            if maj_recall < crit:
                strength *= 0.3

        if self._post_drift_remaining > 0:
            strength *= self.post_drift_aug_reduction

        return max(0.0, float(strength))

    def _augment_sample(self, X, y):
        if self.augmentation_mode == "none":
            return X.copy(), y.copy()

        if self.augmentation_mode == "noise":
            std = self._local_feature_std()
            if std is None:
                return X.copy(), y.copy()

            sigma = self._effective_aug_strength()
            if sigma <= 0.0:
                return X.copy(), y.copy()

            noise = self._rng.normal(loc=0.0, scale=sigma * std, size=X.shape)
            return (X + noise).astype(float), y.copy()

        return X.copy(), y.copy()

    def _construct_new_expert(self):
        idx = self._rng.randint(0, len(self.base_estimators))
        est = cp.deepcopy(self.base_estimators[idx])

        if isinstance(est, (HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier)):
            try:
                est.grace_period = int(self._rng.randint(20, 200))
            except Exception:
                pass

        if isinstance(est, PerceptronMask):
            try:
                est.learning_ratio = float(self._rng.uniform(0.001, 0.05))
            except Exception:
                pass

        weight = 1.0 / max(1, len(self.experts) + 1)
        ex = self.WeightedExpert(est, weight, self.num_classes)

        if self._warmup_samples is not None:
            ex.warmup_remaining = int(self._warmup_samples)

        return ex

    def _update_drift_detector(self, was_correct):
        if not self.enable_drift_detector or self._drift_detector is None:
            return False

        error_value = 0.0 if was_correct else 1.0
        drift = self._drift_detector.update(error_value)
        if drift:
            self._handle_drift()
        return drift

    def _handle_drift(self):
        self._drift_points.append(self._seen_samples)
        self._post_drift_remaining = self.post_drift_cooldown

        majority_class, minority_classes = self._get_majority_and_minorities()

        if self.reset_majority_history_on_drift:
            new_history = deque(maxlen=self.history_buffer_size)
            for Xh, yh in self._history_buffer:
                cls = int(yh[0])
                if cls in minority_classes:
                    new_history.append((Xh, yh))
            self._history_buffer = new_history

        if not self.keep_class_buffers_on_drift:
            self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))
        else:
            new_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))
            for cls, buf in self.class_buffers.items():
                keep_n = max(1, self.class_buffer_size // 2)
                tail = list(buf)[-keep_n:]
                new_buffers[int(cls)] = deque(tail, maxlen=self.class_buffer_size)
            self.class_buffers = new_buffers

    # =============================================================================
    # SOFT ENSEMBLE PREDICTION FOR TRAINING LOOP
    # =============================================================================

    def _soft_ensemble_predict(self, X, expert_probas):
        """
        Soft ensemble prediction from already computed predict_proba values.
        Consistent with predict() and avoids train/test divergence.
        """
        pred_agg = np.zeros(self.num_classes, dtype=float)

        for exp, p in zip(self.experts, expert_probas):
            if p is None:
                continue
            wc = exp.weight_class
            nc = min(len(wc), self.num_classes, p.shape[-1] if len(p.shape) > 1 else 1)
            if len(p.shape) == 2 and p.shape[0] > 0:
                pred_agg[:nc] += p[0, :nc] * wc[:nc]
            elif len(p.shape) == 1:
                pred_agg[:nc] += p[:nc] * wc[:nc]

        return int(np.argmax(pred_agg))

    # =============================================================================
    # SINGLE SAMPLE UPDATE
    # =============================================================================

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        t0 = time.time()

        if self._classes is None and classes is not None:
            self._classes = list(classes)

        if self._classes is not None:
            self.num_classes = max(self.num_classes, len(self._classes))
        if y is not None and len(y) > 0:
            self.num_classes = max(self.num_classes, int(np.max(y)) + 1)

        if self._warmup_samples is None:
            self._warmup_samples = self.warmup_windows * self.period

        for exp in self.experts:
            if len(exp.weight_class) < self.num_classes:
                fill_value = float(np.mean(exp.weight_class) if len(exp.weight_class) else 1.0)
                exp.weight_class = np.pad(
                    exp.weight_class,
                    (0, self.num_classes - len(exp.weight_class)),
                    "constant",
                    constant_values=fill_value,
                )

        true_c = int(y[0])

        self._history_buffer.append((X.copy(), y.copy()))
        self._y_window.append(true_c)

        # Welford update for running std
        self._welford_update(X)

        majority_class, minority_classes = self._get_majority_and_minorities()

        if majority_class is not None and true_c != majority_class:
            self.class_buffers[true_c].append((X.copy(), y.copy()))

        # =============================================================================
        # FAST PREDICTION RETRIEVAL - SOFT VOTING
        # =============================================================================
        expert_preds = []
        expert_probas = []

        for exp in self.experts:
            p = self._safe_predict_proba(exp.estimator, X)
            y_hat = self._safe_predict(exp.estimator, X)

            if p is None and y_hat is not None:
                # Build one-hot proba from hard prediction
                p_fallback = np.zeros((1, self.num_classes), dtype=float)
                c = int(np.clip(int(y_hat[0]), 0, self.num_classes - 1))
                p_fallback[0, c] = 1.0
                p = p_fallback

            if p is not None and p.shape[1] < self.num_classes:
                p = np.pad(p, ((0, 0), (0, self.num_classes - p.shape[1])),
                           "constant", constant_values=0.0)
            elif p is not None and p.shape[1] > self.num_classes:
                p = p[:, :self.num_classes]

            expert_preds.append(y_hat)
            expert_probas.append(p)

        # Soft ensemble prediction consistent with predict()
        ensemble_pred = self._soft_ensemble_predict(X, expert_probas)
        was_correct = (ensemble_pred == true_c)

        # Update sliding window for majority-recall estimate
        self._recent_true.append(true_c)
        self._recent_preds.append(ensemble_pred)

        self._update_drift_detector(was_correct)

        # =============================================================================
        # EXPERT WEIGHT UPDATE
        # =============================================================================
        for i, exp in enumerate(self.experts):
            exp.lifetime += 1
            if exp.warmup_remaining > 0:
                exp.warmup_remaining -= 1

            y_hat = expert_preds[i]
            if y_hat is None or len(y_hat) == 0:
                continue

            wc_len = len(exp.weight_class)
            if wc_len == 0:
                continue
            pred_c = int(np.clip(int(y_hat[0]), 0, wc_len - 1))
            is_minority_sample = (majority_class is not None and true_c != majority_class)

            if pred_c == true_c:
                # Adaptive asymmetry by imbalance ratio:
                # For mild imbalance (ratio < 3): symmetric 0.12/0.10
                # For extreme imbalance (ratio > 8): stronger 0.20/0.08
                # Linear interpolation in between.
                ratio = self._current_imbalance_ratio()
                if ratio <= 3.0:
                    min_rew, maj_rew = 0.12, 0.10
                elif ratio >= 8.0:
                    min_rew, maj_rew = 0.20, 0.08
                else:
                    t = (ratio - 3.0) / 5.0
                    min_rew = 0.12 + t * (0.20 - 0.12)
                    maj_rew = 0.10 + t * (0.08 - 0.10)
                mult = 1.0 + self.beta * (min_rew if is_minority_sample else maj_rew)
            else:
                # Penalty remains symmetric
                mult = 1.0 - self.beta * (0.04 if is_minority_sample else 0.03)

            exp.weight_class[pred_c] = np.clip(exp.weight_class[pred_c] * mult, 1e-4, 1e4)

        # Apply diversity penalty only every 50 steps (speed)
        if not self.enable_diversity and len(self.experts) > 1 and self._seen_samples % 50 == 0:
            type_counts = Counter(e.model_type for e in self.experts)
            for e in self.experts:
                penalty = 1.0 / np.sqrt(type_counts[e.model_type])
                e.weight_class *= penalty

        # =============================================================================
        # LEARN CURRENT SAMPLE
        # =============================================================================
        for exp in self.experts:
            exp.estimator = self.train_model(exp.estimator, X, y, self._classes, sample_weight)

        # =============================================================================
        # REPLAY / AUGMENTATION - BATCHED with regulation
        # =============================================================================
        if self.replay_mode != "off":
            # min_replay_support filter - classes with too few samples
            #     are skipped (replay from 5 samples only adds noise)
            valid_minorities = [
                c for c in minority_classes
                if c in self.class_buffers
                and len(self.class_buffers[c]) >= self.min_replay_support
            ]

            if len(valid_minorities) > 0:
                effective_k = self._effective_replay_k()

                if effective_k > 0:
                    buf_lists = {c: list(self.class_buffers[c]) for c in valid_minorities}

                    X_replay_list = []
                    y_replay_list = []

                    for _ in range(effective_k):
                        c = self._rng.choice(valid_minorities)
                        buf = buf_lists[c]
                        idx = self._rng.randint(0, len(buf))
                        Xr, yr = buf[idx]

                        if self.replay_mode == "augment":
                            Xr, yr = self._augment_sample(Xr, yr)

                        X_replay_list.append(Xr)
                        y_replay_list.append(yr)

                    # One batched partial_fit per expert
                    X_batch = np.vstack(X_replay_list)
                    y_batch = np.concatenate(y_replay_list)

                    for exp in self.experts:
                        exp.estimator = self.train_model(exp.estimator, X_batch, y_batch, self._classes, sample_weight)

        # =============================================================================
        # CLEANUP AND ADDING EXPERTS
        # =============================================================================
        self.experts = [
            e for e in self.experts
            if float(np.sum(e.weight_class)) >= self.theta * self.num_classes
        ]

        while len(self.experts) < self.min_estimators:
            self.experts.append(self._construct_new_expert())

        if len(self.experts) > self.max_estimators:
            sums = [float(np.sum(e.weight_class)) for e in self.experts]
            drop = int(np.argmin(sums))
            self.experts.pop(drop)

        self._seen_samples += 1

        if self._post_drift_remaining > 0:
            self._post_drift_remaining -= 1

        self.update_times.append(time.time() - t0)

        return np.array([ensemble_pred])

    # =============================================================================
    # RESET + PARAMS
    # =============================================================================

    def reset(self):
        self.num_classes = 2
        self._classes = None
        self.experts = []
        self.update_times = deque(maxlen=2000)

        self._y_window = deque(maxlen=self.period)
        self._history_buffer = deque(maxlen=self.history_buffer_size)

        self._warmup_samples = None
        self._seen_samples = 0

        self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))

        self._post_drift_remaining = 0
        self._drift_points = []

        # reset sliding window and Welford
        self._recent_true = deque(maxlen=self.period)
        self._recent_preds = deque(maxlen=self.period)
        self._welford_n = 0
        self._welford_mean = None
        self._welford_M2 = None

        if self.enable_drift_detector:
            self._drift_detector = SimplePageHinkley(
                delta=self.drift_delta,
                threshold=self.drift_threshold,
                alpha=0.999,
                min_detection_interval=self.drift_min_detection_interval,
            )
        else:
            self._drift_detector = None

        for _ in range(self.min_estimators):
            self.experts.append(self._construct_new_expert())

        return self

    def get_params(self, deep=True):
        return {
            "min_estimators": self.min_estimators,
            "max_estimators": self.max_estimators,
            "period": self.period,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "enable_diversity": self.enable_diversity,
            "rwa_strength": self.rwa_strength,
            "use_lifetime_trend": self.use_lifetime_trend,
            "warmup_windows": self.warmup_windows,
            "history_buffer_size": self.history_buffer_size,
            "class_buffer_size": self.class_buffer_size,
            "replay_mode": self.replay_mode,
            "replay_k": self.replay_k,
            "augmentation_mode": self.augmentation_mode,
            "augmentation_strength": self.augmentation_strength,
            "imbalance_aware_augmentation": self.imbalance_aware_augmentation,
            "majority_recall_critical": self.majority_recall_critical,
            "majority_recall_shutoff": self.majority_recall_shutoff,
            "min_replay_support": self.min_replay_support,
            "enable_drift_detector": self.enable_drift_detector,
            "drift_delta": self.drift_delta,
            "drift_threshold": self.drift_threshold,
            "drift_min_detection_interval": self.drift_min_detection_interval,
            "post_drift_cooldown": self.post_drift_cooldown,
            "post_drift_replay_boost": self.post_drift_replay_boost,
            "post_drift_aug_reduction": self.post_drift_aug_reduction,
            "reset_majority_history_on_drift": self.reset_majority_history_on_drift,
            "keep_class_buffers_on_drift": self.keep_class_buffers_on_drift,
            "random_state": self.random_state,
        }
