import numpy as np
from sklearn.linear_model import Lasso
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