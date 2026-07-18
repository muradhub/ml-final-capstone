import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        # 1. Centering the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Computing covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 4. Sorting eigenvalues and eigenvector in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]

        # 5. Store top components and variance ratios
        self.components_ = sorted_eigenvectors[:, :self.n_components].T
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio_ = sorted_eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Project data onto the principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

