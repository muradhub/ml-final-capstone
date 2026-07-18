import numpy as np
import pytest
from src.trees.decision_tree import DecisionTree


def test_binary_classification():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree(max_depth=3, random_state=42)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.array_equal(preds, y)


def test_gini_vs_entropy():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree_gini = DecisionTree(max_depth=3, criterion="gini", random_state=42)
    tree_ent = DecisionTree(max_depth=3, criterion="entropy", random_state=42)
    tree_gini.fit(X, y)
    tree_ent.fit(X, y)
    assert np.array_equal(tree_gini.predict(X), y)
    assert np.array_equal(tree_ent.predict(X), y)


def test_max_depth():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = DecisionTree(max_depth=1, random_state=42)
    tree.fit(X, y)
    assert tree.depth <= 1


def test_min_samples_split():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(min_samples_split=20, random_state=42)
    tree.fit(X, y)
    tree2 = DecisionTree(min_samples_split=2, random_state=42)
    tree2.fit(X, y)
    assert tree.n_leaves <= tree2.n_leaves


def test_predict_proba():
    X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree(max_depth=3, random_state=42)
    tree.fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = tree.predict(X)
    assert np.array_equal(preds, np.argmax(proba, axis=1))


def test_sample_weight():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    y = np.array([0] * 15 + [1] * 15)
    w = np.ones(30)
    w[:15] = 10.0
    tree = DecisionTree(max_depth=2, random_state=42)
    tree.fit(X, y, sample_weight=w)


def test_feature_importances():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(max_depth=3, random_state=42)
    tree.fit(X, y)
    imp = tree.feature_importances()
    assert imp.shape == (5,)
    assert np.allclose(imp.sum(), 1.0)


def test_single_class():
    X = np.random.randn(10, 3)
    y = np.zeros(10, dtype=int)
    tree = DecisionTree(random_state=42)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.all(preds == 0)


def test_multiclass():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    tree = DecisionTree(max_depth=5, random_state=42)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert preds.shape == (60,)


def test_repr():
    tree = DecisionTree()
    assert "unfitted" in repr(tree)
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    y = (X[:, 0] > 0).astype(int)
    tree.fit(X, y)
    r = repr(tree)
    assert "gini=" in r or "DecisionTree" in r


def test_max_features_subset():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(max_features="sqrt", random_state=42)
    tree.fit(X, y)
    tree2 = DecisionTree(max_features="log2", random_state=42)
    tree2.fit(X, y)
    tree3 = DecisionTree(max_features=3, random_state=42)
    tree3.fit(X, y)
    assert tree.feature_importances().shape == (10,)
