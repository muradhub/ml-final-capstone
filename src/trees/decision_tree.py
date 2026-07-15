import numpy as np


class Node:
    def __init__(
            self,
            feature_index: int = None,
            threshold: float = None,
            left: "Node" = None,
            right: "Node" = None,
            value: np.ndarray = None,
            samples: int = 0
    ) -> None:
        """
        Args:
            • feature_index (int): which feature to split on
            • threshold (float): split threshold
            • left, right (Node): child nodes
            • value (np.ndarray of shape [n_classes]): class distribution at leaf; for internal nodes, this
            is the prediction if the tree were truncated at this depth (useful for predict_proba).
            • samples (int): number of training samples that reach this node.
        """
        # Split criteria
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # Data held at node
        self.value = value  # class distribution (weighted counts)
        self.samples = samples  # number of samples reaching this node


class DecisionTree:
    def __init__(
            self,
            max_depth: int = None,
            min_samples_split: int = 2,
            criterion: str = "gini",   # "gini" or "entropy"
            max_features: int | str | None = None,   # For Random Forest
            random_state: int | None = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root = None
        self.classes_ = None
        self._n_features = None
        self._total_impurity_decrease = None
        self._n_samples_total = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> "DecisionTree":
        self.classes_ = np.unique(y)
        self._n_features = X.shape[1]
        self._total_impurity_decrease = np.zeros(self._n_features)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._n_samples_total = X.shape[0]
        self.root = self._build_tree(X, y, sample_weight, depth=0)

        # Normalise feature importances
        total_decrease = np.sum(self._total_impurity_decrease)
        if total_decrease > 0:
            self._total_impurity_decrease /= total_decrease
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        # Computing class distribution for this node
        value = np.array([np.sum(w[y == c]) for c in self.classes_])
        node = Node(value=value, samples=n_samples)
        # Stopping criteria
        is_pure = np.any(value == np.sum(value))
        is_max_depth = self.max_depth is not None and depth >= self.max_depth
        is_too_small = n_samples < self.min_samples_split
        all_x_identical = np.all(X == X[0])

        if is_pure or is_max_depth or is_too_small or all_x_identical:
            return node
        best_feat, best_thresh, best_gain = self._best_split(X, y, w)

        if best_gain > 0:
            left_idx = X[:, best_feat] <= best_thresh
            right_idx = ~left_idx
            # Making recursive calls for building left and right children
            node.feature_index = best_feat
            node.threshold = best_thresh
            node.left = self._build_tree(X[left_idx], y[left_idx], w[left_idx], depth + 1)
            node.right = self._build_tree(X[right_idx], y[right_idx], w[right_idx], depth + 1)

            self._total_impurity_decrease[best_feat] += best_gain * (np.sum(w) / self._n_samples_total)

        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        best_gain = -1
        best_feat = None
        best_thresh = None
        parent_impurity = self._calculate_impurity(y, w)
        total_weight = np.sum(w)
        features_to_check = self._get_features_subset(X.shape[1])
        n_classes = len(self.classes_)

        # Weighted one-hot class indicator, built once per node (not per
        # threshold): class_indicator[i, c] = w[i] if y[i] == classes_[c] else 0.
        class_indicator = np.zeros((len(y), n_classes))
        for c_idx, c in enumerate(self.classes_):
            mask = y == c
            class_indicator[mask, c_idx] = w[mask]
        total_class_weights = class_indicator.sum(axis=0)

        for j in features_to_check:
            # (a) Sort samples by this feature once: O(N log N).
            sort_idx = np.argsort(X[:, j], kind="mergesort")
            x_sorted = X[sort_idx, j]
            class_ind_sorted = class_indicator[sort_idx]

            # Running (cumulative) weighted class counts for the left partition
            # {sorted samples 0..i}. This replaces recomputing impurity from
            # scratch at every threshold with an O(N) linear sweep of running
            # totals, as required by DT.3.c.
            cum_left = np.cumsum(class_ind_sorted, axis=0)

            # (b) A candidate threshold only exists between two samples with
            # distinct feature values (matches the previous np.unique-based
            # threshold set exactly, just without rescanning for it).
            distinct_boundary = x_sorted[:-1] != x_sorted[1:]
            boundary_positions = np.where(distinct_boundary)[0]   # split after index i

            # (c) Vectorized sweep: compute the gain at every candidate
            # threshold in one batched operation instead of looping in
            # Python per threshold (that inner-loop overhead was the
            # remaining bottleneck after switching to cumulative counts).
            if len(boundary_positions) == 0:
                continue
            left_counts = cum_left[boundary_positions]                       # (B, n_classes)
            right_counts = total_class_weights - left_counts                 # (B, n_classes)
            w_left = left_counts.sum(axis=1)                                 # (B,)
            w_right = right_counts.sum(axis=1)                               # (B,)
            valid = (w_left > 0) & (w_right > 0)
            if not np.any(valid):
                continue

            imp_left = self._impurity_from_counts_batch(left_counts[valid], w_left[valid])
            imp_right = self._impurity_from_counts_batch(right_counts[valid], w_right[valid])
            gains = (
                parent_impurity
                - (w_left[valid] / total_weight) * imp_left
                - (w_right[valid] / total_weight) * imp_right
            )

            local_best = np.argmax(gains)
            if gains[local_best] > best_gain:
                best_gain = gains[local_best]
                best_feat = j
                valid_positions = boundary_positions[valid]
                i = valid_positions[local_best]
                best_thresh = (x_sorted[i] + x_sorted[i + 1]) / 2.0
        return best_feat, best_thresh, best_gain

    def _impurity_from_counts_batch(self, class_weights: np.ndarray, total_weight: np.ndarray) -> np.ndarray:
        """Vectorized version of _impurity_from_counts: class_weights is (B, n_classes),
        total_weight is (B,); returns an (B,) array of impurities, one per candidate split.
        Same gini/entropy math as _calculate_impurity, just batched across all thresholds
        for a feature at once so the O(N) sweep in _best_split has no per-threshold
        Python-level loop."""
        p = class_weights / total_weight[:, None]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)
        else:
            # p==0 would make log2(0) = -inf; those terms should contribute 0 to
            # entropy (lim p->0 of p*log2(p) = 0), so substitute a safe value of 1
            # wherever p==0 before taking the log -- log2(1)=0, and it gets
            # multiplied by p=0 anyway, giving the correct zero contribution.
            p_safe = np.where(p > 0, p, 1.0)
            return -np.sum(p * np.log2(p_safe), axis=1)

    def _calculate_impurity(self, y: np.ndarray, w: np.ndarray):
        if len(y) == 0:
            return 0.0
        total_weight = np.sum(w)
        if total_weight == 0.0:
            return 0.0
        p = np.array([np.sum(w[y == c]) for c in self.classes_]) / total_weight
        p = p[p > 0]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        else:
            return -np.sum(p * np.log2(p))

    def _get_features_subset(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, int):
            n_select = self.max_features
        elif self.max_features == "sqrt":
            n_select = int(np.floor(np.sqrt(n_features)))
        elif self.max_features == "log2":
            n_select = int(np.floor(np.log2(n_features)))
        else:
            n_select = n_features
        n_select = int(max(1, min(n_select, n_features)))
        return np.random.choice(n_features, n_select, replace=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.classes_[np.argmax(self._traverse_tree(x, self.root).value)] for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        for x in X:
            leaf_value = self._traverse_tree(x, self.root).value
            probas.append(leaf_value / np.sum(leaf_value))
        return np.array(probas)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Node:
        if node.left is None and node.right is None:
            return node
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    @property
    def depth(self) -> int:
        def get_depth(node):
            if node is None or (node.left is None and node.right is None):
                return 0
            return 1 + max(get_depth(node.left), get_depth(node.right))
        return get_depth(self.root)

    @property
    def n_leaves(self) -> int:
        def get_leaves(node):
            if node is None:
                return 0
            if node.left is None and node.right is None:
                return 1
            return get_leaves(node.left) + get_leaves(node.right)
        return get_leaves(self.root)

    def feature_importances(self) -> np.ndarray:
        return self._total_impurity_decrease

    def __repr__(self) -> str:
        if self.root is None:
            return "DecisionTree(unfitted)"
        if self.depth > 4:
            return f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves}, criterion='{self.criterion}')"

        lines = []

        def describe(node: Node, indent: str) -> None:
            impurity = self._calculate_impurity_from_value(node.value)
            dist = np.array2string(node.value, precision=2, separator=", ")
            if node.left is None and node.right is None:
                lines.append(f"{indent}leaf: {self.criterion}={impurity:.3f}, samples={node.samples}, value={dist}")
                return
            lines.append(
                f"{indent}[X{node.feature_index} <= {node.threshold:.3f}] "
                f"{self.criterion}={impurity:.3f}, samples={node.samples}, value={dist}"
            )
            describe(node.left, indent + "  ")
            describe(node.right, indent + "  ")

        describe(self.root, "")
        return "\n".join(lines)

    def _calculate_impurity_from_value(self, value: np.ndarray) -> float:
        total = np.sum(value)
        if total == 0:
            return 0.0
        p = value[value > 0] / total
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        return -np.sum(p * np.log2(p))
