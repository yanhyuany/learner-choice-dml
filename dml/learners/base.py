from abc import ABC, abstractmethod
import numpy as np

class BaseNuisanceLearner(ABC):
    """
    Abstract base class for all nuisance learners.
    All learners must implement fit() and predict().
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseNuisanceLearner":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_test)