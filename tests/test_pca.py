import numpy as np
import pytest
from src.unsupervised.pca import PCA


def test_pca_fit_transform():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    pca = PCA(n_components=3)
    X_t = pca.fit_transform(X)
    assert X_t.shape == (50, 3)


def test_pca_explained_variance():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    pca = PCA(n_components=5)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    assert len(evr) == 5
    assert np.allclose(evr.sum(), 1.0)
    assert np.all(evr >= 0)


def test_pca_components_shape():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 8)
    pca = PCA(n_components=4)
    pca.fit(X)
    assert pca.components_.shape == (4, 8)


def test_pca_mean():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 6)
    pca = PCA(n_components=2)
    pca.fit(X)
    assert pca.mean_.shape == (6,)


def test_pca_orthogonal_components():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    pca = PCA(n_components=5)
    pca.fit(X)
    comp = pca.components_
    for i in range(comp.shape[0]):
        for j in range(i + 1, comp.shape[0]):
            assert abs(np.dot(comp[i], comp[j])) < 1e-10


def test_pca_transform_centers():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_t = pca.transform(X)
    assert np.allclose(np.mean(X_t, axis=0), 0, atol=1e-10)
