import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from .base import BaseNuisanceLearner


class LassoLearner(BaseNuisanceLearner):
    """
    Lasso regression learner (L1 regularization).
    Default alpha=0.1.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoLearner":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class TunedLassoLearner(BaseNuisanceLearner):
    """
    Lasso with automatic alpha selection via cross-validation (LassoCV).
    Searches over a log-scale grid of alpha values.
    """

    def __init__(self, cv: int = 5):
        self.cv = cv
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedLassoLearner":
        self.model = LassoCV(cv=self.cv)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)