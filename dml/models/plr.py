import numpy as np
from ..utils.cross_fitting import cross_fit
from ..utils.variance import compute_variance, confidence_interval


class PLR:
    """
    Partially Linear Regression via Double/Debiased ML.
    
    Model: Y = theta * D + g(X) + epsilon
    Estimates theta using cross-fitting and partialling-out.
    """

    def __init__(self, learner, n_splits: int = 5, random_state: int = 66):
        self.learner = learner
        self.n_splits = n_splits
        self.random_state = random_state
        self.theta_ = None
        self.var_ = None
        self.ci_ = None

    def fit(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray):
        n = len(Y)

        Y_hat = cross_fit(self.learner, X, Y,
                          n_splits=self.n_splits,
                          random_state=self.random_state)
        Y_tilde = Y - Y_hat

        D_hat = cross_fit(self.learner, X, D,
                          n_splits=self.n_splits,
                          random_state=self.random_state)
        D_tilde = D - D_hat

        self.theta_ = (D_tilde @ Y_tilde) / (D_tilde @ D_tilde)

        psi = D_tilde * (Y_tilde - D_tilde * self.theta_)
        J = np.mean(D_tilde**2)
        self.var_ = compute_variance(psi, J, n)
        self.ci_ = confidence_interval(self.theta_, self.var_)

        return self

    def predict(self):
        if self.theta_ is None:
            raise ValueError("PLR model has not been fit yet. Call fit() first.")
        return {
            "theta": self.theta_,
            "var": self.var_,
            "ci_lower": self.ci_[0],
            "ci_upper": self.ci_[1]
        }