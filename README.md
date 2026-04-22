# Adaptive Models for Classification on Imbalanced Data Streams

This repository contains the source code for the Master's thesis: **Adaptive models for classification on imbalanced data streams**.

The project introduces **IDDCW (Improved DDCW)**, an extended version of the original
Diversified Dynamic Class-Weighted ensemble, designed to handle two major challenges in
stream learning simultaneously:
1. **Concept Drift** (changes in data distribution over time).
2. **Class Imbalance** (significant underrepresentation of specific classes).

IDDCW extends the original DDCW approach with instance-based processing, dynamic per-class
adaptive weighting, minority class replay with Gaussian augmentation, a majority recall
feedback loop, and a Page-Hinkley drift detector.

---

## Project Structure

```text
.
├── data/
│   ├── real/real_raw/           # Place your downloaded raw real-world datasets here
│   ├── real/real_text/          # Raw text datasets (Jigsaw, ElectCovid, FakeNewsComb)
│   ├── real/real_clean/         # Generated preprocessed real datasets
│   ├── synthetic_raw/           # Raw RBF Drift dataset provided by the supervisor
│   ├── synthetic_imbalanced/    # Generated synthetic binary datasets (balanced + 90/10)
│   └── synthetic_multiclass/    # Generated synthetic multiclass datasets
├── model/
│   └── configurable_ddcw.py     # Core implementation of the IDDCW model
├── utils/
│   ├── data_preprocesing.py     # Tabular real-world data preprocessing (MinMax + Label Encoding)
│   ├── drift_metrics.py         # Post-hoc analysis of drift detection (Lag, Recovery)
│   ├── logger.py                # Multiprocessing live terminal dashboard
│   ├── metrics.py               # Calculation of RWA, G-Mean, F1, Minority/Majority recalls
│   ├── model_factory.py         # Model definitions (IDDCW + 5 baselines)
│   └── rwa_metric.py            # RWA (Rarity-Weighted Accuracy) implementation
├── generate_imbalanced_data.py  # Generates synthetic binary and multiclass streams
├── preprocess_rbf.py            # Prepares the supervisor-provided RBF Drift dataset
├── preprocess_fakenews.py       # GloVe + TF-IDF pipeline for ElectCovid and FakeNewsComb
├── preprocess_jigsaw.py         # GloVe + TF-IDF pipeline for Jigsaw Toxic Comments
├── run_experiments_parallel.py  # MAIN SCRIPT: runs all models via multiprocessing
├── generate_plots.py            # Generates comparative plots and LaTeX tables
├── visualize_datasets.py        # Generates drift and class-distribution visualizations
├── analyze_minority_performance.py # Generates confusion matrices and per-class metrics
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Prerequisites & Installation

Due to the dependency on the `scikit-multiflow` framework, this project strictly requires
**Python 3.7.x**. It is highly recommended to use a virtual environment.

**Install required dependencies:**
```bash
pip install -r requirements.txt
```

**Datasets available at:**
https://filebin.net/uq31ml73m89xsuta

---

## 1. Data Preparation

Before running the experiments, you need to generate the synthetic streams and preprocess
the real-world datasets into a unified format (MinMax scaled, last column = integer label).

### Synthetic Binary and Multiclass Datasets
Generates SEA, Agrawal and Hyperplane streams (both balanced and 90/10 imbalanced variants)
plus all multiclass streams:
```bash
python generate_imbalanced_data.py
```

### RBF Drift Dataset
The RBF Drift stream used in the experiments is provided by the thesis supervisor as a
separate raw CSV. Place it at `./data/synthetic_raw/rbf_drift.csv` and run:
```bash
python preprocess_rbf.py
```
This produces `./data/synthetic_imbalanced/rbf_drift_balanced.csv` (~1M samples, 10
features, ~50/50 class ratio, gradual drift preserved).

### Tabular Real-World Datasets
Place your raw files (e.g. `elec.csv`, `kddcup.data_10_percent_corrected`, `airlines.csv`,
`shuttle.trn` + `shuttle.tst`) into `data/real/real_raw/`. Then run:
```bash
python -m utils.data_preprocesing
```

### Text Real-World Datasets (Jigsaw, ElectCovid, FakeNewsComb)
Place raw text CSVs into `./data/real/real_text/` (Jigsaw as `train.csv`, the fake-news
datasets as `elect_covid.csv` and `comb.csv`). Preprocessing uses pretrained GloVe
embeddings and TF-IDF weighting. *Note: the scripts automatically download the required
GloVe models via `gensim.downloader` on the first run.*
```bash
python preprocess_fakenews.py
python preprocess_jigsaw.py
```

---

## 2. Running Experiments

The core execution script is `run_experiments_parallel.py`. It uses Python's
`multiprocessing` to run different seeds and models simultaneously, significantly speeding
up the grid-search evaluation.

**Configuration:**
Open `run_experiments_parallel.py` and adjust the variables near the top of the file if
needed:
* `NUMBER_OF_RUNS = 5` — number of different random seeds per model.
* `DATASET_MODE = "all"` — one of `"synthetic"`, `"real"`, or `"all"`.
* `PRETRAIN_SIZE = 2000` — number of initial samples used to warm up the model before
  prequential evaluation starts. These samples are excluded from metrics.
* `BLOCK_SIZE = 500` — window size for prequential block metrics.

**Start the experiments:**
```bash
python run_experiments_parallel.py
```
A live interactive terminal dashboard shows the progress of each worker, evaluated
samples, and real-time metrics (RWA, G-Mean, Mean Minority Recall, drift detections).

**Outputs:**
All predictions (`.npz`), raw grid results, per-block metrics, and drift-stability stats
are saved into the `results/` directory:
* `grid_search_results_raw.csv` — one row per (dataset × model × run)
* `grid_search_summary.csv` — aggregated across runs
* `prequential_block_metrics.csv` — per-block metrics over time
* `drift_stability_stats.csv` — detection lag, recovery time, post-drift stability
* `predictions/*.npz` — raw `y_true` / `y_pred` / drift points per experiment

---

## 3. Evaluation & Visualization

Once the experiments finish, you can generate analytics, plots, and tables using the
post-processing scripts. All outputs go into `results/figures/` and
`results/minority_analysis/`.

**1. Performance plots and LaTeX tables:**
*(RWA / G-Mean / Macro F1 / Minority Recall over time, showcase comparisons, training
time bar charts, and ready-to-paste LaTeX tables.)*
```bash
python generate_plots.py
```

**2. Dataset characteristics (concept drift & class imbalance):**
*(Feature importance over time + class distribution over time for every dataset.)*
```bash
python visualize_datasets.py
```

**3. Per-class and minority-class analysis:**
*(Confusion matrices and per-class Precision/Recall/F1 extracted from raw predictions;
produces `multiclass_minority_summary.csv` and `worst_cases_by_worst_minority_f1.csv`.)*
```bash
python analyze_minority_performance.py
```

---

## Model Highlights (IDDCW)

The implemented `IDDCW` model introduces several mechanisms for imbalanced drifting
streams:

* **Heterogeneous Ensemble:** A pool of diverse base learners (Naïve Bayes, Hoeffding
  Tree, Hoeffding Adaptive Tree, Extremely Fast Decision Tree, Perceptron). For datasets
  with 50+ features, Naïve Bayes is automatically dropped to avoid numerical overflow.
* **Soft Voting:** Replaces hard majority voting to prevent overconfident
  misclassifications.
* **Adaptive Reward/Penalty:** Per-class weighting dynamically shifts its asymmetry based
  on the current *imbalance ratio (IR)*.
* **Smart Replay & Augmentation:** Minority vzorky sú uložené v cyklických class
  bufferoch a počas tréningu prehrávané. Replay intensity is logarithmically capped by
  the IR and optionally augmented with localized Gaussian noise estimated online via
  Welford's algorithm.
* **Majority Recall Feedback Loop:** Prevents the model from over-focusing on the
  minority class by actively monitoring majority recall. If it drops below adaptive
  thresholds, replay is dynamically throttled or fully shut off.
* **Page-Hinkley Drift Detector:** Operates on the ensemble's error stream with targeted
  post-drift reactions (buffer trimming, temporary replay boost, reduced augmentation
  during cooldown).