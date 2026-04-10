"""
utils/model_factory.py — Definícia a konštrukcia modelov pre experimenty.

Exportované funkcie:
  get_model_name(model) -> (str, dict)   — názov + parametre modelu
  get_model_configs(run_id, n_features)  — zoznam modelov pre jeden beh

Modely (po konzultácii):
  1. DDCW          — hlavný navrhovaný model
  2. ARF            — AdaptiveRandomForest (state-of-the-art streaming ensemble)
  3. OzaBaggingADWIN — OzaBagging + ADWIN drift detektor
  4. LevBag         — LeveragingBagging (silný diverzitný ensemble)
  5. OnlineBoosting — OnlineBoostingClassifier, online AdaBoost (Oza & Russell 2005)
  6. HoeffdingTree  — jednoduchý streaming strom (silný single-model baseline)

Seedy sú odvodené od run_id → reprodukovateľné ale rôzne naprieč behmi.
"""

from skmultiflow.trees import (
    HoeffdingTreeClassifier,
    HoeffdingAdaptiveTreeClassifier,
    ExtremelyFastDecisionTreeClassifier,
)
from skmultiflow.bayes import NaiveBayes
from skmultiflow.neural_networks import PerceptronMask

try:
    from skmultiflow.meta import AdaptiveRandomForestClassifier
    HAS_ARF = True
except Exception:
    HAS_ARF = False

try:
    from skmultiflow.meta import OzaBaggingADWINClassifier
    HAS_OZA = True
except Exception:
    HAS_OZA = False

try:
    from skmultiflow.meta import LeveragingBaggingClassifier
    HAS_LEV = True
except Exception:
    HAS_LEV = False

try:
    from skmultiflow.meta import OnlineBoostingClassifier
    HAS_ADABOOST = True
except Exception:
    HAS_ADABOOST = False

from model.configurable_ddcw_new import Configurable_DDCW


def get_model_name(model):
    if isinstance(model, Configurable_DDCW):
        params = model.get_params()
        name = "DDCW"
        name += f"_mode-{params['replay_mode']}"
        if params["augmentation_mode"] != "none":
            name += f"_aug-{params['augmentation_mode']}{params['augmentation_strength']}"
        if params["enable_drift_detector"]:
            name += "_drift"
        name += "_noDiv" if not params["enable_diversity"] else "_Div"
        name += f"_p{params['period']}"
        name += f"_hb{params['history_buffer_size']}"
        name += f"_cb{params['class_buffer_size']}"
        name += f"_rk{params['replay_k']}"
        return name, params
    return model.__class__.__name__, {}


def get_model_configs(run_id=1, n_features=None):
    """
    Vracia zoznam modelov pre jeden beh experimentu.

    n_features : int alebo None
        Ak >= 50, NaiveBayes sa vynechá z DDCW poolu (overflow pri 54+ príznakoch).
    """
    if n_features is not None and n_features >= 50:
        estimators_hetero = [
            HoeffdingTreeClassifier(grace_period=50),
            HoeffdingTreeClassifier(grace_period=200),
            HoeffdingAdaptiveTreeClassifier(),
            ExtremelyFastDecisionTreeClassifier(),
            PerceptronMask(),
        ]
    else:
        estimators_hetero = [
            NaiveBayes(),
            HoeffdingTreeClassifier(),
            HoeffdingAdaptiveTreeClassifier(),
            ExtremelyFastDecisionTreeClassifier(),
            PerceptronMask(),
        ]

    models = []

    # ── 1) DDCW ───────────────────────────────────────────────────────────────
    models.append(Configurable_DDCW(
        base_estimators=estimators_hetero,
        period=600,
        beta=1.5,
        theta=0.02,
        enable_diversity=False,
        use_lifetime_trend=True,
        history_buffer_size=600,
        class_buffer_size=300,
        replay_mode="augment",
        replay_k=5,
        augmentation_mode="noise",
        augmentation_strength=0.01,
        imbalance_aware_augmentation=True,
        enable_drift_detector=True,
        drift_delta=0.005,
        drift_threshold=100.0,
        drift_min_detection_interval=2000,
        post_drift_cooldown=300,
        post_drift_replay_boost=1,
        post_drift_aug_reduction=0.5,
        reset_majority_history_on_drift=True,
        keep_class_buffers_on_drift=True,
        random_state=400 + run_id,
    ))

    # ── 2) ARF ────────────────────────────────────────────────────────────────
    # State-of-the-art streaming ensemble. Každý strom má vlastný ADWIN detektor.
    if HAS_ARF:
        models.append(AdaptiveRandomForestClassifier(
            n_estimators=10,
            random_state=600 + run_id,
        ))

    # ── 3) OzaBagging ─────────────────────────────────────────────────────────
    # Klasický streaming bagging (Oza & Russell 2001). Každá vzorka sa trénuje
    # k-krát podľa Poisson(1) rozdelenia. Silný všeobecný baseline.
    # ── 3) OzaBagging + ADWIN ─────────────────────────────────────────────────
    # OzaBagging rozšírený o ADWIN drift detektor — konzistentné s ARF.
    if HAS_OZA:
        models.append(OzaBaggingADWINClassifier(
            base_estimator=HoeffdingTreeClassifier(),
            n_estimators=10,
            random_state=700 + run_id,
        ))

    # ── 4) LeveragingBagging ──────────────────────────────────────────────────
    # Silný diverzitný ensemble — leveraging váhy zvyšujú diverzitu členov.
    if HAS_LEV:
        models.append(LeveragingBaggingClassifier(
            base_estimator=HoeffdingTreeClassifier(),
            n_estimators=10,
            random_state=800 + run_id,
        ))

    # ── 5) OnlineBoosting (AdaBoost pre dátové toky) ─────────────────────────
    # Oza & Russell 2005 + ADWIN drift detektor. Vzorky dostávajú váhy podľa
    # chybovosti predchádzajúcich klasifikátorov, simulované cez Poisson(lambda).
    if HAS_ADABOOST:
        models.append(OnlineBoostingClassifier(
            base_estimator=HoeffdingTreeClassifier(),
            n_estimators=10,
            random_state=900 + run_id,
        ))

    # ── 6) HoeffdingTree ─────────────────────────────────────────────────────
    # Jednoduchý single-model baseline. Hoeffding bound zaručuje štatisticky
    # ekvivalentné rozdelenie ako pri neobmedzených dátach.
    models.append(HoeffdingTreeClassifier())

    return models