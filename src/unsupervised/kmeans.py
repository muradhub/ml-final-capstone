import numpy as np


class KMeans:
    def __init__(
            self,
            n_clusters: int,
            max_iter: int = 300,
            tol: float = 1e-4,
            random_state: int | None = None
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.inertia_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "KMeans":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 1. Initialising centroids by picking random data points
        initial_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[initial_idx].copy()

        for i in range(self.max_iter):
            # 2. Computing distances to centroids using broadcasting
            # Distances shape: (n_samples, n_clusters)
            distances = np.sum((X[:, np.newaxis, :] - self.centroids_) ** 2, axis=2)

            # 3. Assigning labels based on closest centroid
            self.labels_ = np.argmin(distances, axis=1)

            # 4. Computing new centroids
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) == 0:
                    new_centroids[k] = X[np.random.choice(n_samples)]
                else:
                    new_centroids[k] = np.mean(cluster_points, axis=0)

            # 5. Checking for convergence (if centroids stopped moving)
            centroid_shift = np.sum((self.centroids_ - new_centroids) ** 2)
            self.centroids_ = new_centroids
            if centroid_shift < self.tol:
                break
        final_distances = np.sum((X[:, np.newaxis, :] - self.centroids_) ** 2, axis=2)
        self.inertia_ = np.sum(np.min(final_distances, axis=1))

        return self
