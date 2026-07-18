import numpy as np
from typing import Iterator
from src.trees.decision_tree import DecisionTree


class DecisionStump(DecisionTree):
    """ Convenience ␣ subclass :␣ max_depth =1 ,␣ single ␣ binary ␣ split ."""
    def __init__(self, criterion: str = "gini", random_state: int | None = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


class AdaBoostClassifier:
    def __init__(
            self,
            n_estimators: int = 50,
            learning_rate: float = 1.0,
            criterion: str = "gini",
            random_state: int | None = None
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        # Storage for trained estimators and weights
        self._estimators = []
        self._estimator_weights = []
        self._estimator_errors = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        N, _ = X.shape
        self.classes_ = np.unique(y)
        w = np.ones(N) / N

        self._estimators = []
        self._estimator_weights = []
        self._estimator_errors = []

        for m in range(self.n_estimators):
            # (i) Fit a stump to the training data weighted by w
            r_state = self.random_state + m if self.random_state is not None else None
            h_m = DecisionStump(criterion=self.criterion, random_state=r_state)
            h_m.fit(X, y, sample_weight=w)

            # Get predictions to calculate error
            predictions = h_m.predict(X)
            misclassified = (predictions != y).astype(int)

            # (ii) Compute weighted error
            epsilon_m = np.sum(w * misclassified) / np.sum(w)

            # (iii) Error clipping and early termination check
            n_classes = len(self.classes_)
            chance_level = (n_classes - 1) / n_classes
            if epsilon_m == 0:
                epsilon_m = 1e-10
            if epsilon_m >= chance_level:
                if m == 0:
                    raise ValueError(f"Early termination triggered: error {epsilon_m:.4f} >= "
                                     f"chance level {chance_level:.4f} (K={n_classes}) at iteration {m}.")
                break
            # (iv) Compute estimator weight (alpha) scaled by learning rate
            alpha_m = self.learning_rate * (np.log((1.0 - epsilon_m) / epsilon_m) + np.log(n_classes - 1))
            # the paper shrinks correct samples while the code inflates incorrect samples.
            w = w * np.exp(alpha_m * misclassified)
            w /= np.sum(w)

            self._estimators.append(h_m)
            self._estimator_weights.append(alpha_m)
            self._estimator_errors.append(epsilon_m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        class_votes = np.zeros((N, len(self.classes_)))
        for alpha, h_m in zip(self._estimator_weights, self._estimators):
            preds = h_m.predict(X)
            for idx, c in enumerate(self.classes_):
                class_votes[:, idx] += alpha * (preds == c)
        return self.classes_[np.argmax(class_votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Documented Approach: Class-wise sum of alpha votes normalized via Softmax
        N = X.shape[0]
        class_votes = np.zeros((N, len(self.classes_)))
        for alpha, h_m in zip(self._estimator_weights, self._estimators):
            preds = h_m.predict(X)
            for idx, c in enumerate(self.classes_):
                class_votes[:, idx] += alpha * (preds == c)
        exp_votes = np.exp(class_votes - np.max(class_votes, axis=1, keepdims=True))
        return exp_votes / np.sum(exp_votes, axis=1, keepdims=True)

    @property
    def estimator_weights(self) -> np.ndarray:
        return np.array(self._estimator_weights)

    @property
    def estimator_errors(self) -> np.ndarray:
        return np.array(self._estimator_errors)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        N = X.shape[0]
        class_votes = np.zeros((N, len(self.classes_)))
        for alpha, h_m in zip(self._estimator_weights, self._estimators):
            preds = h_m.predict(X)
            for idx, c in enumerate(self.classes_):
                class_votes[:, idx] += alpha * (preds == c)

            yield self.classes_[np.argmax(class_votes, axis=1)]
