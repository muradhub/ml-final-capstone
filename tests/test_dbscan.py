import numpy as np
import pytest
from src.unsupervised.dbscan import DBSCAN


def test_dbscan_two_clusters():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(20, 2) * 0.3 + np.array([0, 0]),
        rng.randn(20, 2) * 0.3 + np.array([3, 3]),
    ])
    db = DBSCAN(eps=0.8, min_samples=3)
    db.fit(X)
    n_clusters = len(set(db.labels_) - {-1})
    assert n_clusters == 2


def test_dbscan_noise():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2) * 5.0
    db = DBSCAN(eps=0.3, min_samples=5)
    db.fit(X)
    assert db.noise_fraction_ > 0


def test_dbscan_fit_predict():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(15, 2) * 0.3 + np.array([0, 0]),
        rng.randn(15, 2) * 0.3 + np.array([5, 5]),
    ])
    db = DBSCAN(eps=0.8, min_samples=3)
    labels = db.fit_predict(X)
    assert len(labels) == 30


def test_dbscan_k_distance():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3)
    kd = DBSCAN.k_distance(X, k=3)
    assert kd.shape == (20,)
    assert np.all(kd >= 0)


def test_dbscan_single_cluster():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2) * 0.1 + np.array([0, 0])
    db = DBSCAN(eps=1.0, min_samples=3)
    db.fit(X)
    assert -1 not in db.labels_


def test_dbscan_noise_fraction_property():
    db = DBSCAN(eps=0.5, min_samples=2)
    with pytest.raises(AttributeError):
        _ = db.noise_fraction_


def test_dbscan_all_noise():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2) * 100.0
    db = DBSCAN(eps=0.1, min_samples=5)
    db.fit(X)
    assert np.all(db.labels_ == -1)
    assert db.noise_fraction_ == 1.0
