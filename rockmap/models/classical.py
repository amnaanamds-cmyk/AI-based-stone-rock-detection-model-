"""Classical machine-learning benchmarks: Random Forest and Support Vector Machine."""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ClassicalClassifier:
    """Per-pixel classifier operating on (normalised) feature vectors."""

    def __init__(self, algo: str = "rf", seed: int = 0, n_jobs: int = -1,
                 svm_max_samples: int = 6000):
        if algo not in ("rf", "svm"):
            raise ValueError(f"Unknown classical algorithm: {algo}")
        self.algo = algo
        self.svm_max_samples = svm_max_samples
        if algo == "rf":
            self.model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=n_jobs,
                                                random_state=seed, class_weight="balanced")
        else:
            self.model = SVC(kernel="rbf", C=10.0, gamma="scale", decision_function_shape="ovr",
                             random_state=seed, class_weight="balanced")
        self.train_seconds = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> "ClassicalClassifier":
        if self.algo == "svm" and len(y) > self.svm_max_samples:
            # SVM training is O(n^2); a stratified subsample keeps it tractable.
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), self.svm_max_samples, replace=False)
            x, y = x[idx], y[idx]
        t0 = time.time()
        self.model.fit(x, y)
        self.train_seconds = time.time() - t0
        return self

    @property
    def classes(self) -> np.ndarray:
        return self.model.classes_

    def _proba(self, x: np.ndarray) -> np.ndarray:
        if self.algo == "rf":
            return self.model.predict_proba(x)
        # SVM: soft-max over one-vs-rest decision values gives a pseudo-probability
        d = self.model.decision_function(x)
        e = np.exp(2.0 * (d - d.max(1, keepdims=True)))
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, x: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        out = [self._proba(x[i:i + chunk]) for i in range(0, len(x), chunk)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, len(self.classes)), np.float32)

    def predict_image(self, features: np.ndarray) -> np.ndarray:
        """Normalised (F, H, W) features -> (n_classes, H, W) probabilities."""
        f, h, w = features.shape
        probs = self.predict_proba(features.reshape(f, -1).T)
        return probs.T.reshape(len(self.classes), h, w)

    def feature_importance(self) -> list[float] | None:
        if self.algo == "rf":
            return self.model.feature_importances_.tolist()
        return None

    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "ClassicalClassifier":
        return joblib.load(path)
