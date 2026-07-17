"""
Shared utilities for the experimental study (Section 3 of the project brief).

This module is intentionally free of any experiment-running logic — it only
loads/prepares data and computes/reports metrics. run_all.py imports from here
and orchestrates the actual 7 experiments.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless: never try to open a GUI window on a server/CI box
import matplotlib.pyplot as plt
from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, adjusted_rand_score,
)

RANDOM_SEED = 42
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_breast_cancer_data():
    """Binary, 569 samples, 30 features. sklearn.datasets built-in."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data, data.target, "breast_cancer"


def load_adult_income_data():
    """
    Binary, ~48k samples, severe-ish class imbalance (minority ~24%, not <=1%
    on its own — see make_imbalanced() below, which is applied to whichever
    dataset is flagged as the imbalance case in run_all.py).
    UCI Adult (a.k.a. "Census Income"). Downloaded via OpenML mirror so no
    manual file staging is needed.
    """
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna()
    y = (df["class"] == ">50K").astype(int).values
    X_df = df.drop(columns=["class"])
    X_df = X_df.select_dtypes(include=[np.number])   # drop categoricals; see report note
    return X_df.values.astype(float), y, "adult_income"


def load_covertype_subset_data(n_samples=6000, random_state=RANDOM_SEED):
    """Multi-class, high-dimensional (54 features). Subsampled for tractability."""
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype()
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(data.target), size=min(n_samples, len(data.target)), replace=False)
    return data.data[idx].astype(float), data.target[idx], "covertype_subset"


def load_mnist_binary_data(digits=(0, 1), n_samples=6000, random_state=RANDOM_SEED):
    """High-dimensional (784 features) 2-class MNIST subset."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("mnist_784", version=1, as_frame=False)
    mask = np.isin(data.target.astype(int), digits)
    X, y = data.data[mask], data.target.astype(int)[mask]
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(y), size=min(n_samples, len(y)), replace=False)
    y_bin = (y[idx] == digits[1]).astype(int)
    return X[idx].astype(float), y_bin, "mnist_binary"


DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer_data,
    "adult_income": load_adult_income_data,
    "covertype_subset": load_covertype_subset_data,
    "mnist_binary": load_mnist_binary_data,
}


# --------------------------------------------------------------------------- #
# Preprocessing (spec 3.2)
# --------------------------------------------------------------------------- #

def preprocess(X, y, test_size=0.2, random_state=RANDOM_SEED, impute="drop"):
    """
    - Handles missing values (drop rows by default; see `impute` for mean-fill).
    - 80/20 train/test split.
    - StandardScaler fit on train only, applied to both.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        if impute == "drop":
            X, y = X[~nan_rows], y[~nan_rows]
        elif impute == "mean":
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        else:
            raise ValueError(f"Unknown impute strategy: {impute}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def make_imbalanced(X, y, minority_class=1, target_fraction=0.01, random_state=RANDOM_SEED):
    """
    Subsamples the minority class down to `target_fraction` of the dataset so
    the "severe class imbalance (minority <= 1%)" requirement in spec 3.1 is
    met deliberately and documented, rather than relying on a dataset that
    happens to already be imbalanced.
    """
    rng = np.random.RandomState(random_state)
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]

    n_majority = len(majority_idx)
    n_minority_target = max(1, int(round(target_fraction * n_majority / (1 - target_fraction))))
    if n_minority_target < len(minority_idx):
        minority_idx = rng.choice(minority_idx, size=n_minority_target, replace=False)

    keep = np.concatenate([majority_idx, minority_idx])
    rng.shuffle(keep)
    return X[keep], y[keep]


def apply_class_weighting(y, strategy="inverse_frequency"):
    """
    Returns a per-sample weight array usable as `sample_weight` for our
    from-scratch DecisionTree, as the documented imbalance treatment
    (spec 3.1 "Class Imbalance Treatment") for tree-based models.
    Classes with fewer samples get proportionally larger weight, which is
    equivalent in effect to oversampling but touches the impurity criterion
    directly instead of duplicating rows.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n = len(y)
    if strategy == "inverse_frequency":
        weights = np.array([n / (len(classes) * freq[label]) for label in y])
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    return weights


def flip_labels(y, fraction, random_state=RANDOM_SEED):
    """Randomly flips `fraction` of labels to a different class (noise robustness, exp. 5)."""
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()
    classes = np.unique(y)
    n_flip = int(round(fraction * len(y)))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in flip_idx:
        choices = classes[classes != y_noisy[i]]
        y_noisy[i] = rng.choice(choices)
    return y_noisy


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true, y_pred, y_proba=None, positive_class=None):
    """
    Returns dict(accuracy, f1_macro, auc_roc). AUC is binary if 2 classes are
    present, else macro-averaged one-vs-rest (needs y_proba).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                pos_idx = 1 if positive_class is None else list(classes).index(positive_class)
                metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, pos_idx])
            else:
                metrics["auc_roc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auc_roc"] = float("nan")   # e.g. a fold missing a class
    return metrics


def summarize_cv(fold_metrics: list) -> dict:
    """fold_metrics: list of dicts (one per fold) -> {metric: (mean, std)}"""
    keys = fold_metrics[0].keys()
    return {k: (float(np.mean([m[k] for m in fold_metrics])),
                float(np.std([m[k] for m in fold_metrics]))) for k in keys}


# --------------------------------------------------------------------------- #
# Bias-variance decomposition (spec 3.3, experiment 6; Breiman 1996 definition)
# --------------------------------------------------------------------------- #

def _fit_predict_one_bootstrap(args):
    """
    Module-level (picklable) worker: fit a fresh model on one bootstrap
    resample and return predictions on the fixed test set. Kept at module
    level -- not a closure -- so it works under multiprocessing's 'spawn'
    start method too, not just Linux's 'fork'.
    """
    model_type, model_kwargs, X_boot, y_boot, X_test = args
    if model_type == "tree":
        from src.trees.decision_tree import DecisionTree
        model = DecisionTree(**model_kwargs)
    elif model_type == "adaboost":
        from src.trees.adaboost import AdaBoostClassifier
        model = AdaBoostClassifier(**model_kwargs)
    elif model_type == "random_forest":
        from src.trees.random_forest import RandomForestClassifier
        model = RandomForestClassifier(**model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.fit(X_boot, y_boot)
    return model.predict(X_test)


def bias_variance_decomposition(model_type, model_kwargs, X_train, y_train, X_test, y_test,
                                 n_bootstraps=100, random_state=RANDOM_SEED, n_jobs=1):
    """
    model_type: "tree" | "adaboost" | "random_forest" (dispatched in the
        module-level worker above so this stays picklable for multiprocessing).
    model_kwargs: dict of constructor kwargs for that model, e.g.
        {"max_depth": 5, "random_state": 42}.

    Trains on `n_bootstraps` bootstrap resamples of X_train/y_train, evaluates
    every resulting model once on the fixed X_test/y_test, then computes the
    bias^2 / variance / noise-ish decomposition of 0-1 loss following
    Breiman (1996): for each test point, bias^2 is how often the *main*
    (majority-vote) prediction is wrong, and variance is how often an
    individual model's prediction disagrees with that main prediction.

    n_jobs > 1 farms the (fully independent) bootstrap replicates out to a
    process pool. This changes only *how many CPU cores* do the required
    work concurrently -- the bootstrap indices are generated up front with a
    single seeded RNG regardless of n_jobs, so results are identical to the
    sequential (n_jobs=1) path; only wall-clock time differs.
    """
    rng = np.random.RandomState(random_state)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Bootstrap indices are precomputed sequentially, in the main process,
    # BEFORE any parallel dispatch -- this is what keeps results reproducible
    # regardless of n_jobs / process scheduling order.
    boot_indices = [rng.choice(n_train, n_train, replace=True) for _ in range(n_bootstraps)]

    job_args = [
        (model_type, model_kwargs, X_train[idx], y_train[idx], X_test)
        for idx in boot_indices
    ]

    if n_jobs is not None and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            preds_list = pool.map(_fit_predict_one_bootstrap, job_args)
    else:
        preds_list = [_fit_predict_one_bootstrap(args) for args in job_args]

    all_preds = np.array(preds_list)   # (n_bootstraps, n_test)

    # Main prediction per test point = majority vote across the B models.
    main_pred = np.array([
        np.bincount(all_preds[:, i]).argmax() for i in range(n_test)
    ])

    bias_sq = np.mean(main_pred != y_test)
    variance = np.mean([np.mean(all_preds[:, i] != main_pred[i]) for i in range(n_test)])
    avg_error = np.mean(all_preds != y_test[np.newaxis, :])

    return {"bias_sq": float(bias_sq), "variance": float(variance), "avg_error": float(avg_error)}


# --------------------------------------------------------------------------- #
# Plotting helpers (Agg backend, saved to figures/, never shown interactively)
# --------------------------------------------------------------------------- #

def ensure_figures_dir(dataset_name: str) -> str:
    path = os.path.join(FIGURES_DIR, dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_line_plot(x, ys: dict, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(6, 4))
    for label, y in ys.items():
        plt.plot(x, y, label=label, marker="o", markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter_plot(X_2d, labels, title, out_path, cmap="tab10", legend_title="Label"):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    # tab10 for <=10 groups, tab20 beyond that; colors cycle if there are
    # ever more groups than the palette (shouldn't happen for these datasets).
    palette = plt.get_cmap(cmap if len(unique_labels) <= 10 else "tab20")

    plt.figure(figsize=(6.5, 5))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        if lab == -1:
            # DBSCAN's noise convention -- always gray, always "Noise",
            # regardless of what other cluster ids happen to be present.
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.5,
                        color="lightgray", label="Noise", zorder=1)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.8,
                        color=palette(i % palette.N), label=str(lab), zorder=2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=8, title_fontsize=9, markerscale=1.5, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_bar_plot(labels, values, ylabel, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Timing helper — used to report per-experiment wall time (see run_all.py)
# --------------------------------------------------------------------------- #

class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.time()
        print(f"  -> {self.label} ...", flush=True)
        return self

    def __exit__(self, *exc):
        elapsed = time.time() - self._t0
        print(f"  <- {self.label} done in {elapsed:.1f}s", flush=True)
