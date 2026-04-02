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

def cross_fit_aggregated(
    learner,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_rep: int = 10,
    random_state: int = 66
) -> np.ndarray:
    """
    Aggregated cross-fitting over multiple repetitions.
    Runs cross_fit n_rep times with different fold splits,
    returns the average of all out-of-sample predictions.
    """
    predictions = []
    for i in range(n_rep):
        y_pred = cross_fit(
            learner, X, y,
            n_splits=n_splits,
            random_state=random_state + i
        )
        predictions.append(y_pred)
    return np.mean(predictions, axis=0)

def cross_fit_honest(
    learner_select,
    learner_fit,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 66
) -> np.ndarray:
    """
    Honest cross-fitting with separate data for model selection and estimation.
    
    Splits data into three parts:
    - selection set: tune hyperparameters
    - training set: fit the model
    - test set: predict (out-of-sample)
    """
    n = len(y)
    
    # split off selection set (20% of data)
    n_select = int(n * 0.2)
    X_select, y_select = X[:n_select], y[:n_select]
    X_rest, y_rest = X[n_select:], y[n_select:]
    
    # fit learner_select on selection set to tune
    learner_select.fit(X_select, y_select)
    
    # use learner_fit (same config) on remaining data with cross-fitting
    y_pred_rest = cross_fit(
        learner_fit, X_rest, y_rest,
        n_splits=n_splits,
        random_state=random_state
    )
    
    # fill in predictions (selection set gets mean prediction)
    y_pred = np.zeros(n)
    y_pred[:n_select] = np.mean(y_rest)
    y_pred[n_select:] = y_pred_rest
    
    return y_pred