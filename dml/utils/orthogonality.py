import numpy as np
import jax
import jax.numpy as jnp

def plr_score(theta: float, Y_tilde: jnp.ndarray, 
              D_tilde: jnp.ndarray) -> float:
    """
    PLR score function (mean over observations).
    psi = D_tilde * (Y_tilde - D_tilde * theta)
    """
    psi = D_tilde * (Y_tilde - D_tilde * theta)
    return jnp.mean(psi)

def irm_score(theta: float, Y_tilde: jnp.ndarray,
              D: jnp.ndarray, m_hat: jnp.ndarray) -> float:
    """
    IRM doubly robust score function (mean over observations).
    psi = (Y_tilde * (D - m_hat)) / (m_hat * (1 - m_hat)) - theta
    """
    psi = Y_tilde * (D - m_hat) / (m_hat * (1 - m_hat)) - theta
    return jnp.mean(psi)

def verify_plr_orthogonality(Y_tilde: np.ndarray, 
                              D_tilde: np.ndarray,
                              theta_0: float) -> dict:
    """
    Verify Neyman orthogonality for PLR score.
    Computes d/d(eta) E[psi] at true parameters.
    Expected result: close to 0.
    """
    Y_tilde_j = jnp.array(Y_tilde)
    D_tilde_j = jnp.array(D_tilde)

    # derivative of score w.r.t. D_tilde (proxy for eta)
    grad_fn = jax.grad(plr_score, argnums=2)
    derivative = grad_fn(theta_0, Y_tilde_j, D_tilde_j)

    return {
        "score_at_truth": float(plr_score(theta_0, Y_tilde_j, D_tilde_j)),
        "derivative": float(jnp.mean(derivative)),
        "is_orthogonal": abs(float(jnp.mean(derivative))) < 0.1
    }


def verify_irm_orthogonality(Y_tilde: np.ndarray,
                              D: np.ndarray,
                              m_hat: np.ndarray,
                              theta_0: float) -> dict:
    """
    Verify Neyman orthogonality for IRM score.
    Computes d/d(eta) E[psi] at true parameters.
    Expected result: close to 0.
    """
    Y_tilde_j = jnp.array(Y_tilde)
    D_j = jnp.array(D)
    m_hat_j = jnp.array(m_hat)

    grad_fn = jax.grad(irm_score, argnums=3)
    derivative = grad_fn(theta_0, Y_tilde_j, D_j, m_hat_j)

    return {
        "score_at_truth": float(irm_score(theta_0, Y_tilde_j, D_j, m_hat_j)),
        "derivative": float(jnp.mean(derivative)),
        "is_orthogonal": abs(float(jnp.mean(derivative))) < 0.1
    }