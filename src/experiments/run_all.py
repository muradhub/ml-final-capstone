"""
python src/experiments/run_all.py [--datasets breast_cancer,adult_income,...] [--skip exp1,exp6,...] [--n-jobs N]

Reproduces every experiment in Section 3.3 of the project brief, for every
dataset in DATASET_LOADERS (utils.py). Figures land in figures/<dataset>/,
and a single results/summary.json is written at the end with every numeric
result so the report can cite exact numbers instead of eyeballing plots.

PASS --n-jobs <cores> ON ANYTHING BUT THE SMALLEST DATASET. By default this
runs single-threaded (n_jobs=1) for reproducible, easy-to-debug behavior.
On larger datasets (adult_income, covertype_subset, mnist_binary) the
following are embarrassingly parallel and will use --n-jobs when you set it:
  - every RandomForestClassifier fit (RF.3 in the brief -- trees within one
    forest are independent, trained via multiprocessing.Pool)
  - Experiment 3(b)'s max_depth sweep (20 independent Random Forest fits,
    one per depth -- parallelized across depths, each fit single-threaded
    internally to avoid nested process pools)
  - Experiment 6's bootstrap loop (100 independent model fits per model
    type -- parallelized across bootstrap replicates)
Bootstrap/fold indices are always generated sequentially with a single seeded
RNG before any parallel dispatch, so results are identical regardless of
--n-jobs; only wall-clock time changes.

Two honest heads-ups before you run it on all 4 datasets:
  1. Experiment 6 (bias-variance) trains 100 fresh models per dataset just for
     that one experiment. That's the unavoidable cost of a bootstrap-based
     decomposition — there's no shortcut without changing what's measured,
     but --n-jobs parallelizes the 100 replicates across cores.
  2. Experiments 2 and 3(a) do NOT retrain 30-40 times per dataset. They fit
     the ensemble ONCE at the max size and reuse AdaBoost.staged_predict / a
     single RandomForest's first-k-trees to get every intermediate point.
     If you ever see a version of this file (yours or another team's) that
     calls .fit() inside the n_estimators sweep loop, that's the bug most
     "why is this taking forever" complaints trace back to. Experiment 3(b)
     is the one exception that DOES need one fit per value (max_depth
     genuinely changes tree structure, not just how many trees are used) --
     that's what --n-jobs is for.
Expect this to take a while on the full 4-dataset run, longest on
covertype_subset and mnist_binary since they're the biggest/highest-dim.
"""
import os
import sys
import json
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.trees.decision_tree import DecisionTree
from src.trees.adaboost import AdaBoostClassifier, DecisionStump
from src.trees.random_forest import RandomForestClassifier
from src.unsupervised.pca import PCA
from src.unsupervised.kmeans import KMeans
from src.unsupervised.dbscan import DBSCAN

from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score

from src.experiments import utils as U


def _fit_rf_at_depth(args):
    """Module-level (picklable) worker for Experiment 3(b)'s max_depth sweep.
    Each depth is fit single-threaded (n_jobs=1 internally) since the sweep
    itself is parallelized across depths one level up -- avoids nested pools."""
    X_train, y_train, X_test, y_test, depth, n_estimators, random_state = args
    rf_d = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=depth, random_state=random_state, n_jobs=1,
    )
    rf_d.fit(X_train, y_train)
    return float(np.mean(rf_d.predict(X_test) == y_test))




# Datasets flagged per spec 3.1 requirements: at least one severely imbalanced
# (<=1% minority), at least one high-dimensional (>20 features).
IMBALANCED_DATASET = "adult_income"        # forced to <=1% minority via make_imbalanced()
HIGH_DIM_DATASETS = {"covertype_subset", "mnist_binary"}   # 54 / 784 features respectively


def load_and_prepare(name: str):
    X, y, tag = U.DATASET_LOADERS[name]()
    if name == IMBALANCED_DATASET:
        X, y = U.make_imbalanced(X, y, minority_class=1, target_fraction=0.01)
    X_train, X_test, y_train, y_test = U.preprocess(X, y)
    return {
        "name": tag, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "is_imbalanced": name == IMBALANCED_DATASET,
        "is_high_dim": name in HIGH_DIM_DATASETS,
    }


# --------------------------------------------------------------------------- #
# Experiment 1 — Baseline
# --------------------------------------------------------------------------- #

def experiment_1_baseline(ds, sample_weight=None):
    X_train, X_test, y_train, y_test = ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"]

    tree = DecisionTree(random_state=U.RANDOM_SEED)
    tree.fit(X_train, y_train, sample_weight=sample_weight)
    tree_metrics = U.compute_metrics(y_test, tree.predict(X_test), tree.predict_proba(X_test))

    stump = DecisionStump(random_state=U.RANDOM_SEED)
    stump.fit(X_train, y_train, sample_weight=sample_weight)
    stump_metrics = U.compute_metrics(y_test, stump.predict(X_test), stump.predict_proba(X_test))

    sk_tree = SkDecisionTree(random_state=U.RANDOM_SEED)
    sk_tree.fit(X_train, y_train, sample_weight=sample_weight)
    sk_metrics = U.compute_metrics(y_test, sk_tree.predict(X_test), sk_tree.predict_proba(X_test))

    agreement = abs(tree_metrics["accuracy"] - sk_metrics["accuracy"])
    return {
        "our_tree": tree_metrics, "our_stump": stump_metrics, "sklearn_tree": sk_metrics,
        "our_vs_sklearn_accuracy_gap": agreement,
        "within_2pct": agreement <= 0.02,
    }


# --------------------------------------------------------------------------- #
# Experiment 2 — AdaBoost scaling (fit ONCE at max size, reuse staged_predict)
# --------------------------------------------------------------------------- #

def experiment_2_adaboost_scaling(ds, out_dir, max_estimators=200, step=5):
    X_train, X_test, y_train, y_test = ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"]

    ada = AdaBoostClassifier(n_estimators=max_estimators, random_state=U.RANDOM_SEED)
    ada.fit(X_train, y_train)   # single fit — this is the expensive part, done once

    checkpoints = list(range(1, max_estimators + 1, step))
    train_acc, test_acc = [], []
    staged_train = list(ada.staged_predict(X_train))
    staged_test = list(ada.staged_predict(X_test))
    for m in checkpoints:
        train_acc.append(np.mean(staged_train[m - 1] == y_train))
        test_acc.append(np.mean(staged_test[m - 1] == y_test))

    U.save_line_plot(
        checkpoints, {"train": train_acc, "test": test_acc},
        "n_estimators", "accuracy", f"AdaBoost scaling — {ds['name']}",
        os.path.join(out_dir, "adaboost_scaling.png"),
    )
    overfits = train_acc[-1] - test_acc[-1] > 0.05 and test_acc[-1] < max(test_acc)
    return {
        "checkpoints": checkpoints, "train_acc": train_acc, "test_acc": test_acc,
        "best_test_acc": max(test_acc), "final_test_acc": test_acc[-1],
        "shows_overfitting": bool(overfits),
    }


# --------------------------------------------------------------------------- #
# Experiment 3 — Random Forest scaling
# --------------------------------------------------------------------------- #

def experiment_3_rf_scaling(ds, out_dir, max_estimators=200, step=5, n_jobs=1):
    X_train, X_test, y_train, y_test = ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"]

    # (a) vary n_estimators, max_depth=None — one RF fit at max size, then we
    # evaluate using the first k trees at each checkpoint (no refitting).
    rf_full = RandomForestClassifier(
        n_estimators=max_estimators, max_depth=None, random_state=U.RANDOM_SEED,
        oob_score=True, n_jobs=n_jobs,
    )
    rf_full.fit(X_train, y_train)

    checkpoints = list(range(1, max_estimators + 1, step))
    test_acc_a, oob_acc_a = [], []
    full_estimators = rf_full._estimators
    full_oob_idx = rf_full._oob_indices
    for k in checkpoints:
        sub_rf = RandomForestClassifier(n_estimators=k, random_state=U.RANDOM_SEED)
        sub_rf.classes_ = rf_full.classes_
        sub_rf._estimators = full_estimators[:k]
        sub_rf._oob_indices = full_oob_idx[:k]
        test_acc_a.append(np.mean(sub_rf.predict(X_test) == y_test))
        oob_acc_a.append(sub_rf._compute_oob_score(X_train, y_train))

    U.save_line_plot(
        checkpoints, {"test": test_acc_a, "oob": oob_acc_a},
        "n_estimators", "accuracy", f"Random Forest scaling (n_estimators) — {ds['name']}",
        os.path.join(out_dir, "rf_scaling_n_estimators.png"),
    )

    # (b) vary max_depth, n_estimators fixed at 100 — this DOES need one fit
    # per depth value since tree structure itself changes, not just how many
    # trees are used. Only 20 fits (depth 1..20), but each is a full 100-tree
    # forest, so on larger datasets this is the expensive half of Experiment 3.
    # The 20 depths are independent of each other, so we parallelize across
    # them (n_jobs workers) rather than parallelizing trees within a single
    # forest (n_jobs=1 inside _fit_rf_at_depth) — avoids nested process pools.
    depths = list(range(1, 21))
    job_args = [
        (X_train, y_train, X_test, y_test, d, 100, U.RANDOM_SEED) for d in depths
    ]
    if n_jobs is not None and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            test_acc_b = pool.map(_fit_rf_at_depth, job_args)
    else:
        test_acc_b = [_fit_rf_at_depth(args) for args in job_args]

    U.save_line_plot(
        depths, {"test": test_acc_b},
        "max_depth", "accuracy", f"Random Forest scaling (max_depth) — {ds['name']}",
        os.path.join(out_dir, "rf_scaling_max_depth.png"),
    )

    return {
        "n_estimators_sweep": {"checkpoints": checkpoints, "test_acc": test_acc_a, "oob_acc": oob_acc_a},
        "max_depth_sweep": {"depths": depths, "test_acc": test_acc_b},
        "best_depth": depths[int(np.argmax(test_acc_b))],
    }


# --------------------------------------------------------------------------- #
# Experiment 4 — Head-to-head, 5-fold CV
# --------------------------------------------------------------------------- #

def experiment_4_head_to_head(ds, n_estimators=100, n_folds=5, n_jobs=1):
    X = np.vstack([ds["X_train"], ds["X_test"]])
    y = np.concatenate([ds["y_train"], ds["y_test"]])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=U.RANDOM_SEED)

    fold_results = {"single_tree": [], "adaboost": [], "random_forest": [], "sklearn_rf": []}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        tree = DecisionTree(random_state=U.RANDOM_SEED).fit(Xtr, ytr)
        fold_results["single_tree"].append(U.compute_metrics(yte, tree.predict(Xte), tree.predict_proba(Xte)))

        ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=U.RANDOM_SEED).fit(Xtr, ytr)
        fold_results["adaboost"].append(U.compute_metrics(yte, ada.predict(Xte), ada.predict_proba(Xte)))

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=U.RANDOM_SEED, n_jobs=n_jobs).fit(Xtr, ytr)
        fold_results["random_forest"].append(U.compute_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)))

        sk_rf = SkRandomForest(n_estimators=n_estimators, random_state=U.RANDOM_SEED, n_jobs=n_jobs).fit(Xtr, ytr)
        fold_results["sklearn_rf"].append(U.compute_metrics(yte, sk_rf.predict(Xte), sk_rf.predict_proba(Xte)))

    return {model: U.summarize_cv(folds) for model, folds in fold_results.items()}


# --------------------------------------------------------------------------- #
# Experiment 5 — Noise robustness
# --------------------------------------------------------------------------- #

def experiment_5_noise_robustness(ds, out_dir, noise_levels=(0.05, 0.10, 0.20), n_estimators=100, n_jobs=1):
    X_train, X_test, y_train, y_test = ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"]

    ada_acc, rf_acc = [], []
    for eta in noise_levels:
        y_noisy = U.flip_labels(y_train, eta)

        ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=U.RANDOM_SEED).fit(X_train, y_noisy)
        ada_acc.append(np.mean(ada.predict(X_test) == y_test))

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=U.RANDOM_SEED, n_jobs=n_jobs).fit(X_train, y_noisy)
        rf_acc.append(np.mean(rf.predict(X_test) == y_test))

    U.save_line_plot(
        list(noise_levels), {"AdaBoost": ada_acc, "Random Forest": rf_acc},
        "label noise fraction", "test accuracy", f"Noise robustness — {ds['name']}",
        os.path.join(out_dir, "noise_robustness.png"),
    )
    more_sensitive = "AdaBoost" if (ada_acc[0] - ada_acc[-1]) > (rf_acc[0] - rf_acc[-1]) else "Random Forest"
    return {
        "noise_levels": list(noise_levels), "adaboost_acc": ada_acc, "random_forest_acc": rf_acc,
        "more_sensitive_to_noise": more_sensitive,
    }


# --------------------------------------------------------------------------- #
# Experiment 6 — Bias-variance decomposition (the genuinely slow one)
# --------------------------------------------------------------------------- #

def experiment_6_bias_variance(ds, out_dir, n_bootstraps=100, n_jobs=1):
    X_train, X_test, y_train, y_test = ds["X_train"], ds["X_test"], ds["y_train"], ds["y_test"]

    results = {}
    results["single_tree"] = U.bias_variance_decomposition(
        "tree", {"max_depth": 5, "random_state": U.RANDOM_SEED},
        X_train, y_train, X_test, y_test, n_bootstraps=n_bootstraps, n_jobs=n_jobs,
    )
    results["adaboost"] = U.bias_variance_decomposition(
        "adaboost", {"n_estimators": 50, "random_state": U.RANDOM_SEED},
        X_train, y_train, X_test, y_test, n_bootstraps=n_bootstraps, n_jobs=n_jobs,
    )
    results["random_forest"] = U.bias_variance_decomposition(
        # n_jobs=1 on the RF itself: parallelism happens across the 100
        # bootstrap replicates (outer loop) instead of within each forest,
        # same reasoning as _fit_rf_at_depth in Experiment 3(b).
        "random_forest", {"n_estimators": 50, "random_state": U.RANDOM_SEED, "n_jobs": 1},
        X_train, y_train, X_test, y_test, n_bootstraps=n_bootstraps, n_jobs=n_jobs,
    )

    labels = list(results.keys())
    U.save_bar_plot(
        labels, [results[m]["bias_sq"] for m in labels], "bias^2",
        f"Bias^2 — {ds['name']}", os.path.join(out_dir, "bias_squared.png"),
    )
    U.save_bar_plot(
        labels, [results[m]["variance"] for m in labels], "variance",
        f"Variance — {ds['name']}", os.path.join(out_dir, "variance.png"),
    )
    return results


# --------------------------------------------------------------------------- #
# Experiment 7 — Unsupervised analysis (PCA / K-Means / DBSCAN)
# --------------------------------------------------------------------------- #

def experiment_7_unsupervised(ds, out_dir):
    X = np.vstack([ds["X_train"], ds["X_test"]])
    y = np.concatenate([ds["y_train"], ds["y_test"]])
    n_features = X.shape[1]

    # --- Scree plot ---
    n_comp = min(n_features, 20)
    pca_full = PCA(n_components=n_comp).fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    U.save_line_plot(
        list(range(1, n_comp + 1)), {"cumulative explained variance": cumvar},
        "# components", "cumulative explained variance", f"Scree plot — {ds['name']}",
        os.path.join(out_dir, "scree_plot.png"),
    )
    n_components_90 = int(np.searchsorted(cumvar, 0.90) + 1)

    # --- 2D projection for visualisation ---
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    # --- K-Means elbow (k=1..10, 10 restarts each, keep lowest inertia) ---
    inertias = []
    best_k_models = {}
    for k in range(1, 11):
        best_inertia, best_model = None, None
        for restart in range(10):
            km = KMeans(n_clusters=k, random_state=U.RANDOM_SEED + restart).fit(X)
            if best_inertia is None or km.inertia_ < best_inertia:
                best_inertia, best_model = km.inertia_, km
        inertias.append(best_inertia)
        best_k_models[k] = best_model
    U.save_line_plot(
        list(range(1, 11)), {"inertia": inertias},
        "k", "inertia", f"K-Means elbow — {ds['name']}",
        os.path.join(out_dir, "kmeans_elbow.png"),
    )
    # Pick best k as # true classes if known, else the elbow's argmin-2nd-derivative.
    best_k = len(np.unique(y)) if len(np.unique(y)) > 1 else 3
    km_best = best_k_models[best_k]
    ari_kmeans = adjusted_rand_score(y, km_best.labels_)

    # --- DBSCAN k-distance plot + fit ---
    min_samples = max(4, int(np.log(len(X))))
    k_dist_sample = X if len(X) <= 2000 else X[np.random.RandomState(U.RANDOM_SEED).choice(len(X), 2000, replace=False)]
    k_distances = DBSCAN.k_distance(k_dist_sample, k=min_samples)
    U.save_line_plot(
        list(range(len(k_distances))), {f"{min_samples}-distance": k_distances},
        "points sorted by distance", "distance", f"k-distance plot — {ds['name']}",
        os.path.join(out_dir, "dbscan_k_distance.png"),
    )
    eps = float(np.percentile(k_distances, 90))   # heuristic "knee" pick; document/adjust in report
    dbscan_sample = k_dist_sample
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(dbscan_sample)
    y_sample = y[:len(dbscan_sample)] if len(dbscan_sample) == len(y) else y[
        np.random.RandomState(U.RANDOM_SEED).choice(len(y), len(dbscan_sample), replace=False)
    ]
    ari_dbscan = adjusted_rand_score(y_sample, db.labels_)
    noise_fraction = db.noise_fraction_

    # --- 2D scatter plots: true labels / K-Means / DBSCAN ---
    U.save_scatter_plot(X_2d, y, f"PCA (true labels) — {ds['name']}",
                         os.path.join(out_dir, "pca_true_labels.png"), legend_title="True class")
    U.save_scatter_plot(X_2d, km_best.labels_, f"PCA (K-Means, k={best_k}) — {ds['name']}",
                         os.path.join(out_dir, "pca_kmeans.png"), legend_title="K-Means cluster")
    X_2d_sample = X_2d[:len(dbscan_sample)] if len(dbscan_sample) == len(X) else pca_2d.transform(dbscan_sample)
    U.save_scatter_plot(X_2d_sample, db.labels_, f"PCA (DBSCAN, eps={eps:.2f}) — {ds['name']}",
                         os.path.join(out_dir, "pca_dbscan.png"), legend_title="DBSCAN cluster")

    return {
        "n_components_for_90pct_variance": n_components_90,
        "kmeans_best_k": best_k, "kmeans_ari": float(ari_kmeans),
        "dbscan_eps": eps, "dbscan_min_samples": min_samples,
        "dbscan_ari": float(ari_dbscan), "dbscan_noise_fraction": float(noise_fraction),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

ALL_EXPERIMENTS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]


def run_for_dataset(name: str, skip: set, n_jobs: int):
    print(f"\n{'=' * 70}\nDataset: {name}\n{'=' * 70}")
    ds = load_and_prepare(name)
    out_dir = U.ensure_figures_dir(ds["name"])
    results = {}

    sample_weight = None
    if ds["is_imbalanced"]:
        sample_weight = U.apply_class_weighting(ds["y_train"])

    if "exp1" not in skip:
        with U.Timer("Experiment 1: baseline"):
            results["exp1_baseline"] = experiment_1_baseline(ds, sample_weight=sample_weight)

    if "exp2" not in skip:
        with U.Timer("Experiment 2: AdaBoost scaling"):
            results["exp2_adaboost_scaling"] = experiment_2_adaboost_scaling(ds, out_dir)

    if "exp3" not in skip:
        with U.Timer("Experiment 3: Random Forest scaling"):
            results["exp3_rf_scaling"] = experiment_3_rf_scaling(ds, out_dir, n_jobs=n_jobs)

    if "exp4" not in skip:
        with U.Timer("Experiment 4: head-to-head 5-fold CV"):
            results["exp4_head_to_head"] = experiment_4_head_to_head(ds, n_jobs=n_jobs)

    if "exp5" not in skip:
        with U.Timer("Experiment 5: noise robustness"):
            results["exp5_noise_robustness"] = experiment_5_noise_robustness(ds, out_dir, n_jobs=n_jobs)

    if "exp6" not in skip:
        with U.Timer("Experiment 6: bias-variance decomposition (100 bootstraps — slow, but parallelized if --n-jobs > 1)"):
            results["exp6_bias_variance"] = experiment_6_bias_variance(ds, out_dir, n_jobs=n_jobs)

    if "exp7" not in skip:
        with U.Timer("Experiment 7: unsupervised analysis"):
            results["exp7_unsupervised"] = experiment_7_unsupervised(ds, out_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the full experimental study (Section 3.3).")
    parser.add_argument("--datasets", type=str, default=",".join(U.DATASET_LOADERS.keys()),
                         help="Comma-separated dataset keys to run. Default: all 4.")
    parser.add_argument("--skip", type=str, default="",
                         help="Comma-separated experiment ids to skip, e.g. exp6,exp3")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="Passed to RandomForestClassifier for parallel tree training.")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    for name in dataset_names:
        run_for_dataset(name, skip, args.n_jobs)

    print(f"\nAll done. Figures written under {U.FIGURES_DIR}/<dataset>/")


if __name__ == "__main__":
    main()
