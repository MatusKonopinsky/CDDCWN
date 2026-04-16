# Adaptive Models for Classification on Imbalanced Data Streams

This repository contains the source code for the Master's thesis: **Adaptive models for classification on imbalanced data streams**. 

The project introduces an extended **DDCW (Diversified Dynamic Class-Weighted)** ensemble model designed to handle two major challenges in stream learning simultaneously:
1. **Concept Drift** (changes in data distribution over time).
2. **Class Imbalance** (significant underrepresentation of specific classes).

The model extends the original DDCW approach by introducing instance-based processing, dynamic per-class adaptive weighting, minority class replay with Gaussian augmentation, majority recall feedback loops, and a Page-Hinkley drift detector.

---

## Project Structure

```text
.
├── data/
│   ├── real/real_raw/           # Place your downloaded raw real-world datasets here
│   ├── real/real_clean/         # Generated preprocessed real datasets
│   ├── synthetic_imbalanced/    # Generated synthetic binary datasets
│   └── synthetic_multiclass/    # Generated synthetic multiclass datasets
├── model/
│   └── configurable_ddcw.py     # Core implementation of the extended DDCW model
├── utils/
│   ├── data_preprocesing.py     # Tabular data preprocessing (MinMax, Label Encoding)
│   ├── drift_metrics.py         # Post-hoc analysis of drift detection (Lag, Recovery)
│   ├── logger.py                # Multiprocessing live terminal dashboard
│   ├── metrics.py               # Calculation of AUC, F1, Minority/Majority recalls
│   ├── model_factory.py         # Baseline model configurations
│   └── rwa_metric.py            # RWA (Rarity-Weighted Accuracy) implementation
├── generate_imbalanced_data.py  # Script to generate synthetic streams
├── preprocess_fakenews.py       # GloVe TF-IDF pipeline for Fake News datasets
├── preprocess_jigsaw.py         # GloVe TF-IDF pipeline for Jigsaw Toxic Comments
├── run_experiments_parallel.py  # MAIN SCRIPT: Runs all models via multiprocessing
├── generate_plots.py            # Generates comparative plots and LaTeX tables
├── visualize_datasets.py        # Generates drift and distribution visualizations
├── analyze_minority_performance.py # Generates confusion matrices and per-class metrics
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Prerequisites & Installation

Due to the dependency on the `scikit-multiflow` framework, this project strictly requires **Python 3.7.x**. It is highly recommended to use a virtual environment.

**Install required dependencies:**
```bash
pip install -r requirements.txt
```

---

## 1. Data Preparation

Before running the experiments, you need to generate the synthetic streams and preprocess the real-world datasets into a unified format (MinMax scaled, header-less, target variable in the last column).

### Synthetic Datasets
Generate all synthetic datasets (SEA, Agrawal, Hyperplane, RBF, and Multiclass variants) by running:
```bash
python generate_imbalanced_data.py
```

### Tabular Real-World Datasets
Place your raw CSV/ARFF files (e.g., `elec.csv`, `kddcup.data_10_percent_corrected`, `airlines.csv`, `shuttle.csv`) into `data/real/real_raw/`. Then run:
```bash
python -m utils.data_preprocesing
```

### Text Real-World Datasets (Fake News & Jigsaw)
Text datasets require preprocessing using Pretrained GloVe embeddings and TF-IDF weighting. *Note: The scripts will automatically download the required GloVe models via `gensim.downloader` on the first run.*
```bash
python preprocess_fakenews.py
python preprocess_jigsaw.py
```

---

## 2. Running Experiments

The core execution script is `run_experiments_parallel.py`. It utilizes Python's `multiprocessing` to run different seeds and models simultaneously, significantly speeding up the evaluation.

**Configuration:**
Open `run_experiments_parallel.py` and adjust the variables at the top of the file if needed:
* `NUMBER_OF_RUNS = 5` (Number of different random seeds to evaluate).
* `DATASET_MODE = "real"` (Choose between `"synthetic"`, `"real"`, or `"all"`).

**Start the experiments:**
```bash
python run_experiments_parallel.py
```
*A live interactive terminal dashboard will appear, showing the progress of each worker, evaluated samples, and real-time metrics.*

**Outputs:**
All predictions, raw grid results, and prequential block metrics are automatically saved into the `results/` directory.

---

## 3. Evaluation & Visualization

Once the experiments finish, you can generate comprehensive analytics, plots, and tables using the provided post-processing scripts. All outputs are saved into `results/figures/` and `results/minority_analysis/`.

**1. Generate performance plots and LaTeX tables:**
*(Creates RWA/F1 progress over time, showcase comparisons, and training time bar charts)*
```bash
python generate_plots.py
```

**2. Visualize dataset characteristics (Concept Drift & Imbalance):**
*(Creates plots showing feature importance over time and class distribution changes)*
```bash
python visualize_datasets.py
```

**3. Detailed Minority Class Analysis:**
*(Creates Confusion Matrices and extracts Worst/Mean Minority F1-scores into CSVs)*
```bash
python analyze_minority_performance.py
```

---

## Model Highlights (Extended DDCW)
The implemented `Configurable_DDCW` model introduces several novel mechanics for imbalanced streams:
* **Soft Voting:** Replaces hard majority voting to prevent overconfident misclassifications.
* **Adaptive Reward/Penalty:** Per-class weighting dynamically shifts its asymmetry based on the current *Imbalance Ratio (IR)*.
* **Smart Replay & Augmentation:** Minorities are stored in cyclic buffers and replayed during training. Replay intensity is logarithmically capped by the IR and optionally augmented with localized Gaussian noise.
* **Majority Recall Feedback Loop:** Prevents the model from over-focusing on the minority class by actively monitoring the majority recall. If majority recall drops below adaptive thresholds, replay is dynamically throttled or shut off.
* **Page-Hinkley Drift Detector:** Operates on the ensemble's error stream with targeted post-drift reactions (buffer trimming, temporary replay boosts).
