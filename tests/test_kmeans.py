import numpy as np
import pytest
from src.unsupervised.kmeans import KMeans


def test_kmeans_convergence():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(20, 2) + np.array([0, 0]),
        rng.randn(20, 2) + np.array([5, 5]),
        rng.randn(20, 2) + np.array([10, 0]),
    ])
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)
    assert len(np.unique(km.labels_)) == 3
    assert km.centroids_.shape == (3, 2)
    assert km.inertia_ > 0


def test_kmeans_single_cluster():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    km = KMeans(n_clusters=1, random_state=42)
    km.fit(X)
    assert len(np.unique(km.labels_)) == 1
    assert np.all(km.labels_ == 0)


def test_kmeans_inertia_decreases():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(20, 2) + np.array([0, 0]),
        rng.randn(20, 2) + np.array([5, 5]),
    ])
    km = KMeans(n_clusters=2, max_iter=300, random_state=42)
    km.fit(X)
    assert km.inertia_ > 0
    km2 = KMeans(n_clusters=2, max_iter=1, random_state=42)
    km2.fit(X)
    assert km.inertia_ <= km2.inertia_ + 1e-6


def test_kmeans_labels_assignment():
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(10, 2) + np.array([0, 0]),
        rng.randn(10, 2) + np.array([100, 100]),
    ])
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(X)
    assert all(km.labels_[:10] == km.labels_[0])
    assert all(km.labels_[10:] == km.labels_[10])
