import numpy as np
from multiprocessing import Pool
from src.trees.decision_tree import DecisionTree


def _train_one_tree(args):
    """Module-level helper so it can be pickled by multiprocessing.Pool."""
    X_boot, y_boot, tree_params, r_state = args
    tree = DecisionTree(random_state=r_state, **tree_params)
    tree.fit(X_boot, y_boot)
    return tree


class RandomForestClassifier:
    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int | None = None,
            max_features: int | str = "sqrt",   # int, "sqrt", "log2", or None
            min_samples_split: int = 2,
            bootstrap: bool = True,
            oob_score: bool = False,
            n_jobs: int = 1,   # parallelism with multiprocessing
            random_state: int | None = None
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Storage for trained estimators
        self._estimators = []
        self._oob_indices = []   # OOB sample indices per tree, used for OOB scoring
        self.classes_ = None
        self._oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        tree_params = dict(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
        )

        # RF.1 Bootstrap sampling: draw N samples with replacement per tree,
        # and record the out-of-bag (OOB) indices (samples never drawn).
        boot_samples = []
        self._oob_indices = []
        for t in range(self.n_estimators):
            if self.bootstrap:
                boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            else:
                boot_idx = np.arange(n_samples)
            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx, assume_unique=False)
            boot_samples.append((X[boot_idx], y[boot_idx]))
            self._oob_indices.append(oob_idx)

        # RF.2 / RF.3 Train each tree, sequentially or in parallel.
        # random_state for each tree is derived from the forest seed and tree index
        # so results are reproducible regardless of n_jobs.
        r_states = [
            (self.random_state + t) if self.random_state is not None else None
            for t in range(self.n_estimators)
        ]
        job_args = [
            (boot_samples[t][0], boot_samples[t][1], tree_params, r_states[t])
            for t in range(self.n_estimators)
        ]

        if self.n_jobs is not None and self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                self._estimators = pool.map(_train_one_tree, job_args)
        else:
            # Sequential baseline (also used when n_jobs == 1)
            self._estimators = [_train_one_tree(args) for args in job_args]

        # RF.4 Out-of-bag score
        if self.oob_score:
            self._oob_score_ = self._compute_oob_score(X, y)

        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        oob_votes = np.zeros((n_samples, n_classes))
        voted_at_least_once = np.zeros(n_samples, dtype=bool)

        for tree, oob_idx in zip(self._estimators, self._oob_indices):
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            for idx, c in enumerate(self.classes_):
                oob_votes[oob_idx, idx] += (preds == c)
            voted_at_least_once[oob_idx] = True

        # Samples that were never OOB for any tree are excluded from the score.
        scorable = voted_at_least_once
        oob_pred = self.classes_[np.argmax(oob_votes[scorable], axis=1)]
        return float(np.mean(oob_pred == y[scorable]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # RF.5: average the per-tree probability vectors (soft voting).
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes))
        for tree in self._estimators:
            tree_proba = np.zeros((n_samples, n_classes))
            raw_proba = tree.predict_proba(X)
            # Align each tree's own class ordering (tree.classes_) with the forest's.
            for idx, c in enumerate(tree.classes_):
                forest_idx = np.searchsorted(self.classes_, c)
                tree_proba[:, forest_idx] = raw_proba[:, idx]
            proba_sum += tree_proba
        return proba_sum / len(self._estimators)

    @property
    def oob_score_(self) -> float:
        if not self.oob_score:
            raise AttributeError("oob_score_ is only available when oob_score=True was set before fit().")
        return self._oob_score_

    @property
    def feature_importances_(self) -> np.ndarray:
        # RF.6: average the feature importances of all trees.
        importances = np.array([tree.feature_importances() for tree in self._estimators])
        return np.mean(importances, axis=0)
