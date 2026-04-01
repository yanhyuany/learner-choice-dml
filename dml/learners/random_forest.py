import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base import BaseNuisanceLearner

class RandomForestLearner(BaseNuisanceLearner):
    """
    Random Forest regression learner.
    Default n_estimators=100, random_state=42.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestLearner":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)