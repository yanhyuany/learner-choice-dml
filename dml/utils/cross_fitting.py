import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple

def cross_fit(
    learner,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 66
) -> np.ndarray:
    """
    Run K-fold cross-fitting and return out-of-sample predictions.
    
    For each fold, fit the learner on the remaining K-1 folds and
    predict on the held-out fold. This ensures every prediction is
    made on data the model hasn't seen, avoiding overfitting bias.
    """
    n = len(y)
    y_pred = np.zeros(n)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        y_pred[test_idx] = learner.fit_predict(X_train, y_train, X_test)
    
    return y_pred