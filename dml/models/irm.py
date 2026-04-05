import numpy as np
from sklearn.model_selection import KFold
from ..utils.variance import compute_variance, confidence_interval


class IRM:
    """
    Interactive Regression Model via Double/Debiased ML.

    Estimates ATE using the doubly robust score from Chernozhukov et al. (2018).
    D must be binary (0 or 1).

    Score:
        psi_b = g(1,X) - g(0,X)
                + D * (Y - g(1,X)) / m(X)
                - (1-D) * (Y - g(0,X)) / (1 - m(X))
        psi_a = -1
    """

    def __init__(self, learner, n_splits: int = 5, random_state: int = 66,
                 trim: float = 0.01):
        self.learner = learner
        self.n_splits = n_splits
        self.random_state = random_state
        self.trim = trim

        self.theta_ = None
        self.var_ = None
        self.ci_ = None

    def fit(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray):
        n = len(Y)
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)

        # step 1: cross-fit m(X) = P(D=1|X)
        m_hat = np.zeros(n)
        for train_idx, test_idx in kf.split(X):
            self.learner.fit(X[train_idx], D[train_idx])
            m_hat[test_idx] = self.learner.predict(X[test_idx])
        m_hat = np.clip(m_hat, self.trim, 1 - self.trim)

        # step 2: cross-fit g(0,X) = E[Y|D=0, X]
        g0_hat = np.zeros(n)
        for train_idx, test_idx in kf.split(X):
            idx0 = train_idx[D[train_idx] == 0]
            self.learner.fit(X[idx0], Y[idx0])
            g0_hat[test_idx] = self.learner.predict(X[test_idx])

        # step 3: cross-fit g(1,X) = E[Y|D=1, X]
        g1_hat = np.zeros(n)
        for train_idx, test_idx in kf.split(X):
            idx1 = train_idx[D[train_idx] == 1]
            self.learner.fit(X[idx1], Y[idx1])
            g1_hat[test_idx] = self.learner.predict(X[test_idx])

        # step 4: doubly robust ATE score (Chernozhukov et al. 2018)
        psi_b = (g1_hat - g0_hat
                 + D * (Y - g1_hat) / m_hat
                 - (1 - D) * (Y - g0_hat) / (1 - m_hat))
        psi_a = -np.ones(n)
        self.theta_ = -np.mean(psi_b) / np.mean(psi_a)

        # step 5: variance and CI
        psi = psi_b + psi_a * self.theta_
        J = np.mean(psi_a)
        self.var_ = compute_variance(psi, J, n)
        self.ci_ = confidence_interval(self.theta_, self.var_)

        return self

    def predict(self):
        if self.theta_ is None:
            raise ValueError("IRM model has not been fit yet. Call fit() first.")
        return {
            "theta": self.theta_,
            "var": self.var_,
            "ci_lower": self.ci_[0],
            "ci_upper": self.ci_[1]
        }