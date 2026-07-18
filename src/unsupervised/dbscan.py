import numpy as np


class DBSCAN:
    def __init__(self, eps: float, min_samples: int) -> None:
        """
        Args:
            eps (float): neighbourhood radius.
            min_samples (int): number of points (including the point itself)
                required within eps for a point to be considered a core point.
        """
        self.eps = eps
        self.min_samples = min_samples

        self.labels_ = None   # cluster ids; -1 = noise

    def fit(self, X: np.ndarray) -> "DBSCAN":
        n_samples = X.shape[0]
        # UNVISITED = -2 (sentinel), NOISE = -1, cluster ids start at 0.
        self.labels_ = np.full(n_samples, -2, dtype=int)

        neighbors = self._region_query_all(X)

        cluster_id = 0
        for i in range(n_samples):
            if self.labels_[i] != -2:
                continue   # already visited

            if len(neighbors[i]) < self.min_samples:
                # Not (yet) a core point; provisionally mark as noise.
                # It may later be claimed as a border point by a neighbouring cluster.
                self.labels_[i] = -1
                continue

            # i is a core point: expand a new cluster from it.
            self._expand_cluster(i, neighbors, cluster_id)
            cluster_id += 1

        return self

    def _region_query_all(self, X: np.ndarray) -> list:
        """Precompute, for every point, the indices of points within eps (inclusive)."""
        n_samples = X.shape[0]
        dist = self._pairwise_dist(X)
        within_eps = dist <= self.eps
        return [np.where(within_eps[i])[0] for i in range(n_samples)]

    @staticmethod
    def _pairwise_dist(X: np.ndarray, row_block: int = 500) -> np.ndarray:
        """Memory-safe pairwise Euclidean distances, shape (N, N).

        Naively broadcasting X[:, None, :] - X[None, :, :] materializes an
        (N, N, D) tensor -- for N=2000, D=784 (e.g. mnist_binary) that's
        23.4 GiB and crashes. Instead we use the identity
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        which only ever needs an (N, N) matrix, independent of D. Rows are
        processed in blocks so peak memory stays O(row_block * N) rather
        than O(N^2), which also keeps this safe if N grows in the future.
        """
        sq_norms = np.sum(X ** 2, axis=1)
        n = X.shape[0]
        dist = np.empty((n, n), dtype=float)
        for start in range(0, n, row_block):
            end = min(start + row_block, n)
            block = X[start:end]
            sq_dist_block = sq_norms[start:end, None] + sq_norms[None, :] - 2.0 * block @ X.T
            np.maximum(sq_dist_block, 0, out=sq_dist_block)  # clip tiny negative rounding error
            dist[start:end] = np.sqrt(sq_dist_block)
        return dist

    def _expand_cluster(self, seed_idx: int, neighbors: list, cluster_id: int) -> None:
        self.labels_[seed_idx] = cluster_id
        seeds = list(neighbors[seed_idx])

        pos = 0
        while pos < len(seeds):
            q = seeds[pos]
            pos += 1

            if self.labels_[q] == -1:
                # Was marked noise, but is density-reachable: claim as a border point.
                self.labels_[q] = cluster_id
            if self.labels_[q] != -2:
                # Already assigned to this or another cluster during this expansion.
                continue

            self.labels_[q] = cluster_id
            if len(neighbors[q]) >= self.min_samples:
                # q is itself a core point: grow the neighbourhood search.
                for r in neighbors[q]:
                    if self.labels_[r] == -2 or self.labels_[r] == -1:
                        seeds.append(r)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    @staticmethod
    def k_distance(X: np.ndarray, k: int) -> np.ndarray:
        """
        Helper for the required k-distance plot (spec 2.4.3): for each point,
        compute the distance to its k-th nearest neighbour, sorted ascending.
        Not part of the binding interface, but needed to choose eps near the "knee".
        """
        dist = DBSCAN._pairwise_dist(X)
        dist_sorted = np.sort(dist, axis=1)
        # Column 0 is distance to self (0.0); the k-th nearest neighbour is column k.
        k_dist = dist_sorted[:, k]
        return np.sort(k_dist)

    @property
    def noise_fraction_(self) -> float:
        if self.labels_ is None:
            raise AttributeError("Call fit() before accessing noise_fraction_.")
        return float(np.mean(self.labels_ == -1))
