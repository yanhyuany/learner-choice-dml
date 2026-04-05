import numpy as np
import jax
import jax.numpy as jnp


# ── PLR Score ────────────────────────────────────────────────────────────────

def plr_score_full(theta: float, Y: jnp.ndarray, D: jnp.ndarray,
                   l_hat: jnp.ndarray, m_hat: jnp.ndarray) -> float:
    """
    PLR partialling-out score (Score 4.4 in Chernozhukov et al. 2018).
    psi = (D - m(X)) * (Y - l(X) - theta * (D - m(X)))
    """
    D_tilde = D - m_hat
    Y_tilde = Y - l_hat
    psi = D_tilde * (Y_tilde - theta * D_tilde)
    return jnp.mean(psi)


def verify_plr_orthogonality(Y: np.ndarray, D: np.ndarray,
                              l_hat: np.ndarray, m_hat: np.ndarray,
                              theta_0: float,
                              h_scale: float = 0.1,
                              n_directions: int = 10,
                              random_seed: int = 42) -> dict:
    """
    Verify Neyman orthogonality for PLR score.
    Checks both nuisance directions: l(X) and m(X).
    Each nuisance uses independent random perturbation directions.
    """
    Y_j = jnp.asarray(Y, dtype=jnp.float32)
    D_j = jnp.asarray(D, dtype=jnp.float32)
    l_j = jnp.asarray(l_hat, dtype=jnp.float32)
    m_j = jnp.asarray(m_hat, dtype=jnp.float32)
    n = len(Y)
    rng = np.random.default_rng(random_seed)

    derivs_l, derivs_m = [], []

    for _ in range(n_directions):
        # independent direction for each nuisance
        h_l_np = rng.normal(size=n) * h_scale
        h_l_np = h_l_np - h_l_np.mean()
        h_l = jnp.array(h_l_np, dtype=jnp.float32)

        h_m_np = rng.normal(size=n) * h_scale
        h_m_np = h_m_np - h_m_np.mean()
        h_m = jnp.array(h_m_np, dtype=jnp.float32)

        def phi_l(t):
            return plr_score_full(theta_0, Y_j, D_j, l_j + t * h_l, m_j)

        def phi_m(t):
            return plr_score_full(theta_0, Y_j, D_j, l_j, m_j + t * h_m)

        derivs_l.append(float(jax.grad(phi_l)(0.0)))
        derivs_m.append(float(jax.grad(phi_m)(0.0)))

    score_val = float(plr_score_full(theta_0, Y_j, D_j, l_j, m_j))

    return {
        "score_at_estimated_nuisance": score_val,
        "mean_abs_deriv_wrt_l": float(np.mean(np.abs(derivs_l))),
        "mean_abs_deriv_wrt_m": float(np.mean(np.abs(derivs_m))),
        "max_abs_deriv_wrt_l":  float(np.max(np.abs(derivs_l))),
        "max_abs_deriv_wrt_m":  float(np.max(np.abs(derivs_m))),
        "max_abs_derivative":   float(max(np.max(np.abs(derivs_l)),
                                          np.max(np.abs(derivs_m)))),
        "note": "Small directional derivatives provide numerical "
                "support for orthogonality."
    }


# ── IRM Score ────────────────────────────────────────────────────────────────

def irm_score_full(theta: float, g0: jnp.ndarray, g1: jnp.ndarray,
                   m_hat: jnp.ndarray, Y: jnp.ndarray,
                   D: jnp.ndarray) -> float:
    """
    IRM doubly robust ATE score (Chernozhukov et al. 2018, Section 5.1).
    """
    m_safe = jnp.clip(m_hat, 1e-3, 1 - 1e-3)
    psi = (g1 - g0
           + D * (Y - g1) / m_safe
           - (1 - D) * (Y - g0) / (1 - m_safe)
           - theta)
    return jnp.mean(psi)


def verify_irm_orthogonality(g0: np.ndarray, g1: np.ndarray,
                              m_hat: np.ndarray, Y: np.ndarray,
                              D: np.ndarray,
                              theta_0: float,
                              h_scale: float = 0.01,
                              n_directions: int = 10,
                              random_seed: int = 42) -> dict:
    """
    Verify Neyman orthogonality for IRM score.
    Checks all three nuisance directions: g0, g1, m(X).
    Each nuisance uses independent random perturbation directions.
    """
    g0_j = jnp.asarray(g0, dtype=jnp.float32)
    g1_j = jnp.asarray(g1, dtype=jnp.float32)
    m_j  = jnp.asarray(m_hat, dtype=jnp.float32)
    Y_j  = jnp.asarray(Y, dtype=jnp.float32)
    D_j  = jnp.asarray(D, dtype=jnp.float32)
    n = len(Y)
    rng = np.random.default_rng(random_seed)

    derivs_g0, derivs_g1, derivs_m = [], [], []

    for _ in range(n_directions):
        # independent direction for each nuisance
        h_g0_np = rng.normal(size=n) * h_scale
        h_g0_np = h_g0_np - h_g0_np.mean()
        h_g0 = jnp.array(h_g0_np, dtype=jnp.float32)

        h_g1_np = rng.normal(size=n) * h_scale
        h_g1_np = h_g1_np - h_g1_np.mean()
        h_g1 = jnp.array(h_g1_np, dtype=jnp.float32)

        h_m_np = rng.normal(size=n) * h_scale
        h_m_np = h_m_np - h_m_np.mean()
        h_m = jnp.array(h_m_np, dtype=jnp.float32)

        def phi_g0(t):
            return irm_score_full(theta_0, g0_j + t * h_g0,
                                  g1_j, m_j, Y_j, D_j)

        def phi_g1(t):
            return irm_score_full(theta_0, g0_j,
                                  g1_j + t * h_g1, m_j, Y_j, D_j)

        def phi_m(t):
            return irm_score_full(theta_0, g0_j, g1_j,
                                  m_j + t * h_m, Y_j, D_j)

        derivs_g0.append(float(jax.grad(phi_g0)(0.0)))
        derivs_g1.append(float(jax.grad(phi_g1)(0.0)))
        derivs_m.append(float(jax.grad(phi_m)(0.0)))

    score_val = float(irm_score_full(theta_0, g0_j, g1_j, m_j, Y_j, D_j))

    return {
        "score_at_estimated_nuisance": score_val,
        "mean_abs_deriv_wrt_g0": float(np.mean(np.abs(derivs_g0))),
        "mean_abs_deriv_wrt_g1": float(np.mean(np.abs(derivs_g1))),
        "mean_abs_deriv_wrt_m":  float(np.mean(np.abs(derivs_m))),
        "max_abs_deriv_wrt_g0":  float(np.max(np.abs(derivs_g0))),
        "max_abs_deriv_wrt_g1":  float(np.max(np.abs(derivs_g1))),
        "max_abs_deriv_wrt_m":   float(np.max(np.abs(derivs_m))),
        "max_abs_derivative":    float(max(np.max(np.abs(derivs_g0)),
                                           np.max(np.abs(derivs_g1)),
                                           np.max(np.abs(derivs_m)))),
        "note": "Small directional derivatives provide numerical "
                "support for orthogonality."
    }