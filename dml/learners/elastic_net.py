import numpy as np
from sklearn.linear_model import ElasticNet
from .base import BaseNuisanceLearner

class ElasticNetLearner(BaseNuisanceLearner):
    """
    ElasticNet regression learner (L1 + L2 regularization).
    Default alpha=0.1, l1_ratio=0.5.
    """

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetLearner":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)