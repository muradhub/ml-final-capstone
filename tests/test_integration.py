"""Integration tests tying together the full experimental pipeline."""
import numpy as np
import pytest
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.trees.decision_tree import DecisionTree
from src.trees.adaboost import AdaBoostClassifier
from src.trees.random_forest import RandomForestClassifier
from src.unsupervised.pca import PCA
from src.unsupervised.kmeans import KMeans
from src.unsupervised.dbscan import DBSCAN
from src.experiments import utils as U


def test_end_to_end_trees():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    dt = DecisionTree(max_depth=5, random_state=42)
    dt.fit(X, y)
    dt_pred = dt.predict(X)

    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    ada_pred = ada.predict(X)

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=1)
    rf.fit(X, y)
    rf_pred = rf.predict(X)

    assert dt_pred.shape == y.shape
    assert ada_pred.shape == y.shape
    assert rf_pred.shape == y.shape


def test_end_to_end_unsupervised():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 6)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    assert X_2d.shape == (80, 2)

    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)
    assert len(np.unique(km.labels_)) == 3

    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(X)
    assert db.labels_.shape == (80,)


def test_experiment_1_baseline_runs():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 4)
    y = (X[:, 0] > 0).astype(int)
    X_train, X_test, y_train, y_test = U.preprocess(X, y, test_size=0.25, random_state=42)
    ds = {"name": "test", "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    from src.experiments.run_all import experiment_1_baseline
    result = experiment_1_baseline(ds)
    assert "our_tree" in result
    assert "sklearn_tree" in result
    assert result["within_2pct"]


def test_experiment_7_unsupervised_runs():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    y = (X[:, 0] > 0).astype(int)
    X_train, X_test, y_train, y_test = U.preprocess(X, y, test_size=0.2, random_state=42)
    ds = {"name": "test", "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    from src.experiments.run_all import experiment_7_unsupervised
    with tempfile.TemporaryDirectory() as tmp:
        result = experiment_7_unsupervised(ds, tmp)
        assert "n_components_for_90pct_variance" in result
        assert "kmeans_ari" in result
        assert "dbscan_ari" in result
