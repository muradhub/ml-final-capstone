import numpy as np
import pytest
from src.trees.adaboost import AdaBoostClassifier, DecisionStump


def test_adaboost_binary():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    preds = ada.predict(X)
    assert np.array_equal(preds, y)


def test_adaboost_predict_proba():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ada = AdaBoostClassifier(n_estimators=5, random_state=42)
    ada.fit(X, y)
    proba = ada.predict_proba(X)
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_adaboost_staged_predict():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    stages = list(ada.staged_predict(X))
    assert len(stages) == 10
    for s in stages:
        assert s.shape == (6,)
    assert np.array_equal(stages[-1], y)


def test_decision_stump():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    stump = DecisionStump(random_state=42)
    stump.fit(X, y)
    assert stump.depth <= 1


def test_estimator_weights():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ada = AdaBoostClassifier(n_estimators=5, random_state=42)
    ada.fit(X, y)
    ew = ada.estimator_weights
    assert len(ew) > 0
    ee = ada.estimator_errors
    assert len(ee) > 0
    assert np.all(ee > 0)


def test_multiclass_adaboost():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    preds = ada.predict(X)
    assert preds.shape == (60,)


def test_adaboost_single_feature():
    X = np.array([[1], [2], [3], [10], [11], [12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    preds = ada.predict(X)
    assert np.array_equal(preds, y)
