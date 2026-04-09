import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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


class TunedRandomForestLearner(BaseNuisanceLearner):
    """
    Random Forest with GridSearchCV over max_depth and max_features.
    Reference: Bach et al. (2024).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedRandomForestLearner":
        param_grid = {
            'max_depth': [4, 5, 6],
            'max_features': [0.3, 0.5, 0.7],
        }
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)