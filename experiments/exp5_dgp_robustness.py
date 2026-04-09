import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dml.learners.lasso import TunedLassoLearner
from dml.learners.random_forest import TunedRandomForestLearner
from dml.learners.neural_net import TunedNeuralNetLearner
from dml.models.plr import PLR

# -----------------------------------------------------------------------
# DGP Generators
# -----------------------------------------------------------------------

def dgp_sparse_linear(n_obs: int, alpha: float = 0.5,
                      random_state: int = None):
    """
    DGP 1: High-dimensional sparse linear — favorable for Lasso.
    p=100, only first 5 variables have signal, linear nuisance.
    """
    np.random.seed(random_state)
    p = 100
    X = np.random.randn(n_obs, p)
    coef = np.zeros(p)
    coef[:5] = [1.0, 0.8, 0.6, 0.4, 0.2]
    g0 = X @ coef
    m0 = X @ (coef * 0.5)
    D = m0 + np.random.randn(n_obs)
    Y = alpha * D + g0 + np.random.randn(n_obs)
    return X, Y, D


def dgp_piecewise(n_obs: int, alpha: float = 0.5,
                  random_state: int = None):
    """
    DGP 2: Piecewise / step function — favorable for RandomForest.
    p=20, nuisance functions are combinations of indicator functions.
    """
    np.random.seed(random_state)
    p = 20
    X = np.random.randn(n_obs, p)
    g0 = (2.0 * (X[:, 0] > 0).astype(float)
          + 1.0 * (X[:, 1] > 0).astype(float)
          + 0.5 * (X[:, 2] > 0).astype(float))
    m0 = ((X[:, 0] > 0).astype(float)
          + 0.5 * (X[:, 1] > 0).astype(float))
    D = m0 + np.random.randn(n_obs)
    Y = alpha * D + g0 + np.random.randn(n_obs)
    return X, Y, D


def dgp_nonlinear_interaction(n_obs: int, alpha: float = 0.5,
                               random_state: int = None):
    """
    DGP 3: Complex nonlinear interactions — favorable for NeuralNet.
    p=20, nuisance functions include variable interactions and smooth nonlinearities.
    """
    np.random.seed(random_state)
    p = 20
    X = np.random.randn(n_obs, p)
    g0 = np.sin(X[:, 0] * X[:, 1]) + np.exp(-X[:, 2] ** 2)
    m0 = np.tanh(X[:, 0] + X[:, 1]) + 0.3 * X[:, 2]
    D = m0 + np.random.randn(n_obs)
    Y = alpha * D + g0 + np.random.randn(n_obs)
    return X, Y, D


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

DGP_GENERATORS = {
    "Sparse_Linear": dgp_sparse_linear,
    "Piecewise": dgp_piecewise,
    "Nonlinear_Interaction": dgp_nonlinear_interaction,
}

LEARNERS = {
    "Lasso": TunedLassoLearner,
    "RandomForest": TunedRandomForestLearner,
    "NeuralNet": TunedNeuralNetLearner,
}

N_VALUES = [200, 500, 1000]
ALPHA = 0.5
N_REPS = 100


# -----------------------------------------------------------------------
# Single rep
# -----------------------------------------------------------------------

def run_single_rep(learner_class, dgp_fn, n_obs: int, alpha: float,
                   random_state: int) -> dict:
    X, Y, D = dgp_fn(n_obs, alpha, random_state)
    plr = PLR(learner=learner_class(), n_splits=5, random_state=random_state)
    plr.fit(Y, D, X)
    results = plr.predict()

    theta = results['theta']
    covered = results['ci_lower'] < alpha < results['ci_upper']

    return {
        "theta": theta,
        "covered": covered,
        "bias": theta - alpha,
        "rmse": (theta - alpha) ** 2,
    }


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def run_experiment_5(n_reps: int = N_REPS) -> pd.DataFrame:
    """
    Exp 5: DGP robustness using tuned learners.
    3 DGPs × 3 tuned learners × n values × n_reps.
    """
    records = []

    for dgp_name, dgp_fn in DGP_GENERATORS.items():
        for learner_name, learner_class in LEARNERS.items():
            for n_obs in N_VALUES:
                print(f"[{dgp_name}] [{learner_name}] n={n_obs}...")
                biases, rmses, covered = [], [], []

                for rep in range(n_reps):
                    try:
                        res = run_single_rep(
                            learner_class, dgp_fn, n_obs, ALPHA,
                            random_state=rep
                        )
                        biases.append(res['bias'])
                        rmses.append(res['rmse'])
                        covered.append(res['covered'])
                    except Exception:
                        continue

                records.append({
                    "dgp": dgp_name,
                    "learner": learner_name,
                    "n_obs": n_obs,
                    "bias": np.mean(biases),
                    "rmse": np.sqrt(np.mean(rmses)),
                    "coverage": np.mean(covered),
                    "n_valid": len(biases),
                })
                print(f"  bias={np.mean(biases):.4f}, "
                      f"rmse={np.sqrt(np.mean(rmses)):.4f}, "
                      f"coverage={np.mean(covered):.3f}")

    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_experiment_5(df: pd.DataFrame, save_path: str = None):
    dgp_names = list(DGP_GENERATORS.keys())
    metrics = [
        ("rmse", "RMSE"),
        ("coverage", "Coverage Rate"),
        ("bias", "Bias"),
    ]

    learner_colors = {
        "Lasso": "steelblue",
        "RandomForest": "green",
        "NeuralNet": "purple",
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row, dgp_name in enumerate(dgp_names):
        df_dgp = df[df["dgp"] == dgp_name]

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            for learner_name, color in learner_colors.items():
                subset = df_dgp[
                    (df_dgp["learner"] == learner_name) &
                    (df_dgp["n_obs"].isin(N_VALUES))
                ]
                ax.plot(subset["n_obs"], subset[metric],
                        marker='o', label=learner_name, color=color)

            if metric == "bias":
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            if metric == "coverage":
                ax.axhline(y=0.95, color='black', linestyle='--',
                           linewidth=1.5)
                ax.set_ylim([0.5, 1.0])

            ax.set_xlabel('Sample size (n)')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dgp_name}\n{ylabel}")
            ax.legend(fontsize=7)
            ax.set_xscale('log')

    plt.suptitle('Exp 5: DGP Robustness (Tuned Learners)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_summary_table(df: pd.DataFrame, n_obs: int = 500,
                       save_path: str = None):
    df_n = df[df["n_obs"] == n_obs]
    pivot = df_n.pivot(index="dgp", columns="learner", values="rmse")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='RMSE')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}",
                    ha='center', va='center', fontsize=10)

    ax.set_title(f'RMSE Summary — Tuned Learners (n={n_obs})\nGreen = better, Red = worse')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()