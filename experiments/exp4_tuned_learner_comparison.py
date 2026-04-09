import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from dml.learners.lasso import LassoLearner, TunedLassoLearner
from dml.learners.random_forest import RandomForestLearner, TunedRandomForestLearner
from dml.learners.neural_net import NeuralNetLearner, TunedNeuralNetLearner
from dml.models.plr import PLR

# baseline learners (Exp 2)
LEARNERS_BASELINE = {
    "Lasso": LassoLearner,
    "RandomForest": RandomForestLearner,
    "NeuralNet": NeuralNetLearner,
}

# tuned learners (Exp 4)
LEARNERS_TUNED = {
    "Lasso_tuned": TunedLassoLearner,
    "RandomForest_tuned": TunedRandomForestLearner,
    "NeuralNet_tuned": TunedNeuralNetLearner,
}

N_VALUES = [200, 500, 1000, 2000]
ALPHA = 0.5


def generate_data(n_obs: int, alpha: float = ALPHA, random_state: int = None):
    np.random.seed(random_state)
    X, Y, D = make_plr_CCDDHNR2018(
        alpha=alpha, n_obs=n_obs, dim_x=20, return_type='array'
    )
    return X, Y, D


def run_single_rep(learner_class, n_obs: int, alpha: float,
                   random_state: int) -> dict:
    X, Y, D = generate_data(n_obs, alpha, random_state)
    plr = PLR(learner=learner_class(), n_splits=5, random_state=random_state)
    plr.fit(Y, D, X)
    results = plr.predict()

    theta = results['theta']
    ci_lower = results['ci_lower']
    ci_upper = results['ci_upper']
    covered = ci_lower < alpha < ci_upper

    return {
        "theta": theta,
        "covered": covered,
        "bias": theta - alpha,
        "rmse": (theta - alpha) ** 2
    }


def run_experiment_4(n_reps: int = 500,
                     learners: dict = None) -> pd.DataFrame:
    if learners is None:
        learners = LEARNERS_TUNED

    records = []
    for learner_name, learner_class in learners.items():
        for n_obs in N_VALUES:
            print(f"[{learner_name}] n={n_obs}...")
            biases, rmses, covered = [], [], []

            for rep in range(n_reps):
                try:
                    res = run_single_rep(learner_class, n_obs, ALPHA,
                                        random_state=rep)
                    biases.append(res['bias'])
                    rmses.append(res['rmse'])
                    covered.append(res['covered'])
                except Exception:
                    continue

            records.append({
                "learner": learner_name,
                "n_obs": n_obs,
                "bias": np.mean(biases),
                "rmse": np.sqrt(np.mean(rmses)),
                "coverage": np.mean(covered),
                "n_valid": len(biases)
            })
            print(f"  bias={np.mean(biases):.4f}, "
                  f"rmse={np.sqrt(np.mean(rmses)):.4f}, "
                  f"coverage={np.mean(covered):.3f}")

    return pd.DataFrame(records)


def plot_experiment_4(df_baseline: pd.DataFrame,
                      df_tuned: pd.DataFrame,
                      save_path: str = None):
    """
    Plot baseline vs tuned comparison side by side.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    learner_colors = {
        "Lasso": "steelblue",
        "RandomForest": "green",
        "NeuralNet": "purple",
        "Lasso_tuned": "steelblue",
        "RandomForest_tuned": "green",
        "NeuralNet_tuned": "purple",
    }

    metrics = [
        ("bias", "Bias", "Bias (θ̂ - θ₀)"),
        ("rmse", "RMSE", "RMSE"),
        ("coverage", "CI Coverage Rate", "Coverage Rate"),
    ]

    for col, (metric, title, ylabel) in enumerate(metrics):
        for row, (df, label) in enumerate([(df_baseline, "Baseline (default)"),
                                            (df_tuned, "Tuned")]):
            ax = axes[row, col]
            for learner_name in df["learner"].unique():
                color = learner_colors.get(learner_name, "gray")
                # clean name for legend
                display_name = learner_name.replace("_tuned", "")
                subset = df[df["learner"] == learner_name]
                ax.plot(subset["n_obs"], subset[metric],
                        marker='o', label=display_name, color=color)

            if metric == "bias":
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            if metric == "coverage":
                ax.axhline(y=0.95, color='black', linestyle='--',
                           linewidth=1.5, label='Target: 95%')
                ax.set_ylim([0.5, 1.0])

            ax.set_xlabel('Sample size (n)')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {label}")
            ax.legend(fontsize=8)
            ax.set_xscale('log')

    plt.suptitle('Exp 4: Baseline vs Tuned Learner Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()