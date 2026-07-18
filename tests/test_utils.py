import numpy as np
import pytest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experiments import utils as U


def test_compute_metrics_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    m = U.compute_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in m
    assert "f1_macro" in m
    assert "auc_roc" in m
    assert 0 <= m["accuracy"] <= 1


def test_compute_metrics_no_proba():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = U.compute_metrics(y_true, y_pred)
    assert "auc_roc" not in m or m["auc_roc"] is None


def test_summarize_cv():
    folds = [
        {"accuracy": 0.9, "f1_macro": 0.89},
        {"accuracy": 0.92, "f1_macro": 0.91},
        {"accuracy": 0.88, "f1_macro": 0.87},
    ]
    s = U.summarize_cv(folds)
    assert "accuracy" in s
    assert s["accuracy"][0] == pytest.approx(0.9)
    assert len(s["accuracy"]) == 2


def test_make_imbalanced():
    X = np.random.randn(1000, 3)
    y = np.array([0] * 500 + [1] * 500)
    Xb, yb = U.make_imbalanced(X, y, minority_class=1, target_fraction=0.05)
    assert np.mean(yb == 1) <= 0.06


def test_apply_class_weighting():
    y = np.array([0, 0, 0, 1, 1])
    w = U.apply_class_weighting(y)
    assert w.shape == (5,)
    assert w[0] == w[1] == w[2]
    assert w[3] == w[4]
    assert w[0] < w[3]


def test_flip_labels():
    y = np.array([0, 0, 0, 1, 1, 1])
    y_noisy = U.flip_labels(y, fraction=0.5, random_state=42)
    n_flipped = np.sum(y != y_noisy)
    assert n_flipped > 0


def test_preprocess():
    X = np.random.randn(100, 5)
    y = np.array([0] * 50 + [1] * 50)
    Xtr, Xte, ytr, yte = U.preprocess(X, y, test_size=0.2, random_state=42)
    assert len(Xtr) == 80
    assert len(Xte) == 20
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Xtr.std(axis=0), 1, atol=1e-10)


def test_preprocess_drop_nan():
    X = np.random.randn(100, 3)
    X[0, 0] = np.nan
    y = np.zeros(100)
    Xtr, Xte, ytr, yte = U.preprocess(X, y)
    assert len(Xtr) + len(Xte) >= 99


def test_timer():
    with U.Timer("test") as t:
        pass
    assert hasattr(t, '_t0')


def test_bias_variance_decomposition():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    y = (X[:, 0] > 0).astype(int)
    X_train, X_test, y_train, y_test = U.preprocess(X, y, test_size=0.3, random_state=42)
    result = U.bias_variance_decomposition(
        "tree", {"max_depth": 3, "random_state": 42},
        X_train, y_train, X_test, y_test, n_bootstraps=5, n_jobs=1,
    )
    assert "bias_sq" in result
    assert "variance" in result
    assert "avg_error" in result


def test_dataset_loaders():
    for name, loader in U.DATASET_LOADERS.items():
        X, y, tag = loader()
        assert X.shape[0] == len(y)
        assert tag == name
        break
