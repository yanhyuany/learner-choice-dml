import numpy as np

def compute_variance(psi: np.ndarray, J: float, n: int) -> float:
    """
    Compute variance of theta_hat using the influence function.
    
    psi : influence function values, shape (n,)
    J   : E[D_tilde^2], the Jacobian term
    n   : sample size
    """
    return (1 / J**2) * np.mean(psi**2) / n


def confidence_interval(theta: float, var: float, alpha: float = 0.05):
    """
    Compute two-sided confidence interval.
    Returns (lower, upper).
    """
    z = 1.96  # 95% CI
    se = np.sqrt(var)
    return (theta - z * se, theta + z * se)