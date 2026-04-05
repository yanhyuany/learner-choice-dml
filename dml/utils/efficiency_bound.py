import numpy as np
import jax
import jax.numpy as jnp

def plr_efficiency_bound(Y_tilde: np.ndarray,
                         D_tilde: np.ndarray,
                         theta: float) -> dict:
    """
    Compute semiparametric efficiency bound for PLR.
    
    V* = E[D_tilde^2]^{-1} * E[epsilon^2] * E[D_tilde^2]^{-1}
    where epsilon = Y_tilde - theta * D_tilde
    """
    n = len(Y_tilde)
    epsilon = Y_tilde - theta * D_tilde
    J = np.mean(D_tilde ** 2)
    sigma2 = np.mean(epsilon ** 2)
    V_star = sigma2 / (J ** 2) / n
    return {
        "efficiency_bound": V_star,
        "J": J,
        "sigma2": sigma2
    }

def compare_learner_efficiency(Y_tilde: np.ndarray,
                               D_tilde: np.ndarray,
                               theta: float,
                               actual_var: float) -> dict:
    """
    Compare a learner's actual variance against the efficiency bound.
    A ratio close to 1 means the learner is near-efficient.
    """
    bound = plr_efficiency_bound(Y_tilde, D_tilde, theta)
    V_star = bound["efficiency_bound"]
    ratio = actual_var / V_star

    return {
        "efficiency_bound": V_star,
        "actual_variance": actual_var,
        "efficiency_ratio": ratio,
        "is_efficient": ratio < 1.5
    }

def plr_efficiency_bound_jax(Y_tilde: np.ndarray,
                              D_tilde: np.ndarray,
                              theta: float) -> dict:
    """
    Compute efficiency bound using JAX for automatic differentiation.
    Uses JAX to compute the Jacobian J = E[d psi/d theta].
    """
    Y_j = jnp.array(Y_tilde)
    D_j = jnp.array(D_tilde)

    def score(theta):
        psi = D_j * (Y_j - D_j * theta)
        return jnp.mean(psi)

    # negate to match numpy convention: J = E[D_tilde^2] > 0
    J = -float(jax.grad(score)(theta))
    epsilon = Y_tilde - theta * D_tilde
    sigma2 = float(jnp.mean(jnp.array(epsilon) ** 2))
    n = len(Y_tilde)
    V_star = sigma2 / (J ** 2) / n

    return {
        "efficiency_bound": V_star,
        "J_jax": J,
        "sigma2": sigma2
    }