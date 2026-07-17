# ML Final Capstone — Ensemble Methods: Boosting vs. Bagging

Final project for the Machine Learning module, AI Academy, National AI Center — Spring 2026.

Implements Decision Trees (CART), AdaBoost (SAMME) with decision stumps, and Random
Forest from scratch, then compares boosting vs. bagging across four datasets, backed by
a from-scratch unsupervised pipeline (PCA, K-Means, DBSCAN).

## Structure

```
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- trees/
|   |   |-- decision_tree.py      # CART decision tree
|   |   |-- adaboost.py           # AdaBoost (SAMME) with stumps
|   |   '-- random_forest.py      # Random Forest classifier
|   |-- unsupervised/
|   |   |-- pca.py                # Principal Component Analysis
|   |   |-- kmeans.py             # K-Means clustering
|   |   '-- dbscan.py             # DBSCAN clustering
|   '-- experiments/
|       |-- run_all.py            # Orchestrates all 7 experiments
|       '-- utils.py              # Data loading, metrics, plotting
|-- tests/                        # Unit & integration tests
|-- notebooks/
|   '-- analysis.ipynb            # Results summary notebook
|-- figures/                      # Generated experiment plots
|-- data/
|   '-- download_data.sh          # Data download notes
|-- results/
|   '-- summary.json              # All numeric results (created by run_all.py)
|-- report/
|   |-- report.tex                # LaTeX report (IEEE format)
|   '-- report.pdf                # Compiled report
|-- presentation/
|   '-- presentation.pdf          # Slide deck
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pandas` is required even though no `src/` module imports it directly: `sklearn`'s
`fetch_openml(..., as_frame=True)` (used to load the Adult Income dataset) needs it
internally and raises `ImportError` at runtime without it. It's pinned in
`requirements.txt` for that reason.

## Running Experiments

```bash
python src/experiments/run_all.py                          # all 4 datasets, all 7 experiments
python src/experiments/run_all.py --datasets breast_cancer  # single dataset
python src/experiments/run_all.py --skip exp6,exp3          # skip specific experiment(s)
python src/experiments/run_all.py --n-jobs 4                # parallel RF training
```

`--datasets` and `--skip` combine, which is useful for re-running just one piece
without paying for the full run again, e.g. after fixing a bug that only affected one
dataset/experiment:

```bash
python src/experiments/run_all.py --datasets mnist_binary --skip exp1,exp2,exp3,exp4,exp5,exp6 --n-jobs 4
```

**Runtime.** A full run across all 4 datasets takes on the order of an hour or more —
this is expected, not a bug. Experiment 6 (100 bootstrap replicates) and the Random
Forest sweeps in Experiments 3–5 dominate the time, especially on `adult_income` (~49k
rows) and `mnist_binary` (784 features). Use `--n-jobs` to parallelize, and `--skip`
while iterating on one experiment so you're not re-running the whole suite.

**First run needs internet access.** All four datasets are fetched on demand via
`sklearn.datasets` / OpenML on first use (see `data/download_data.sh` for exact
sources); there's no manual download step, but the machine you run this on needs
outbound network access at least once.

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # coverage report
```

Current coverage: **78%** overall (rubric requires ≥60%), all 58 tests passing.
Lowest-covered module is `run_all.py` at 41%, which is expected — most of its
uncovered lines are plotting/orchestration code exercised by full experiment runs
rather than unit tests, not core algorithm logic.

## Experiments

1. Baseline comparison (our tree vs. sklearn)
2. AdaBoost scaling (1–200 estimators)
3. Random Forest scaling (estimators & depth sweeps)
4. Head-to-head 5-fold CV
5. Noise robustness (5% / 10% / 20% label noise)
6. Bias–variance decomposition (100 bootstraps)
7. Unsupervised analysis (PCA, K-Means, DBSCAN)

## Datasets

| Dataset | Size | Task | Source |
|---|---|---|---|
| Breast Cancer Wisconsin | 569 × 30 | Binary | `sklearn.datasets.load_breast_cancer` |
| Adult Income | ~49k × 14 | Binary, imbalanced | `fetch_openml("adult", version=2)` |
| Covertype Subset | 6k × 54 | 7-class | `fetch_covtype`, subsampled |
| MNIST Binary (0 vs. 1) | 6k × 784 | Binary, high-dim | `fetch_openml("mnist_784")`, subsampled |

AdaBoost is implemented as general K-class SAMME (not restricted to binary), so it
runs on `covertype_subset` directly rather than needing a one-vs-rest wrapper — see
the report's methods section for the reasoning.

## Dependencies

- `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `scipy` — core / data loading / plotting
- `pytest`, `pytest-cov` — testing and coverage
- `jupyter`, `nbformat` — notebook (development only)

`scikit-learn` is used only for dataset loading and as the sanity-check baseline
(`sklearn.tree.DecisionTreeClassifier`, etc.) per the project brief — it is never
used as the primary implementation of any of the three supervised models.

## License

MIT
