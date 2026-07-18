import numpy as np
import pytest
from src.trees.random_forest import RandomForestClassifier


def test_rf_binary():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=1)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert np.array_equal(preds, y)


def test_rf_predict_proba():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_rf_oob_score():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=42, n_jobs=1)
    rf.fit(X, y)
    score = rf.oob_score_
    assert 0.0 <= score <= 1.0


def test_rf_feature_importances():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    assert imp.shape == (5,)


def test_rf_no_bootstrap():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=42, n_jobs=1)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (30,)


def test_rf_multiclass():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (60,)


def test_rf_single_estimator():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    rf = RandomForestClassifier(n_estimators=1, random_state=42, n_jobs=1)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert np.array_equal(preds, y)


def test_rf_parallel():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=2)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (30,)
