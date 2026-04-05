import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from dml.learners.lasso import LassoLearner
from dml.learners.elastic_net import ElasticNetLearner
from dml.learners.random_forest import RandomForestLearner
from dml.learners.neural_net import NeuralNetLearner
from dml.learners.causal_forest import CausalForestLearner
from dml.models.plr import PLR
from dml.utils.variance import compute_variance, confidence_interval

def generate_data(n_obs: int = 500, alpha: float = 0.5,
                  random_state: int = None):
    np.random.seed(random_state)
    X, Y, D = make_plr_CCDDHNR2018(
        alpha=alpha, n_obs=n_obs, dim_x=20, return_type='array'
    )
    return X, Y, D


def estimate_nonorthogonal(X, Y, D, learner):
    g_hat = learner.fit(X, Y).predict(X)
    Y_res = Y - g_hat
    theta = np.sum(D * Y_res) / np.sum(D * D)
    psi = D * (Y_res - D * theta)
    J = np.mean(D ** 2)
    var = np.mean(psi ** 2) / (J ** 2) / len(Y)
    se = np.sqrt(var)
    return theta, se


def estimate_dml_no_split(X, Y, D, learner):
    # use separate instances to avoid state contamination
    learner_y = learner.__class__()
    learner_d = learner.__class__()
    Y_hat = learner_y.fit(X, Y).predict(X)
    D_hat = learner_d.fit(X, D).predict(X)
    D_tilde = D - D_hat
    Y_tilde = Y - Y_hat
    theta = (D_tilde @ Y_tilde) / (D_tilde @ D_tilde)
    psi = D_tilde * (Y_tilde - D_tilde * theta)
    J = np.mean(D_tilde ** 2)
    var = np.mean(psi ** 2) / (J ** 2) / len(Y)
    se = np.sqrt(var)
    return theta, se


def estimate_dml_crossfit(X, Y, D, learner):
    plr = PLR(learner=learner, n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    return results['theta'], np.sqrt(results['var'])


def run_single_rep(learner_instance, n_obs: int, alpha: float,
                   random_state: int) -> dict:
    X, Y, D = generate_data(n_obs, alpha, random_state)

    t1, s1 = estimate_nonorthogonal(X, Y, D, learner_instance())
    t2, s2 = estimate_dml_no_split(X, Y, D, learner_instance())
    t3, s3 = estimate_dml_crossfit(X, Y, D, learner_instance())

    r1 = (t1 - alpha) / s1 if s1 > 0 else np.nan
    r2 = (t2 - alpha) / s2 if s2 > 0 else np.nan
    r3 = (t3 - alpha) / s3 if s3 > 0 else np.nan

    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3):
        return {"nonorth": r1, "nosplit": r2, "crossfit": r3}
    return None


def run_experiment_1(learner_name: str, learner_class,
                     n_obs: int = 500, alpha: float = 0.5,
                     n_reps: int = 500) -> dict:
    results = {"nonorth": [], "nosplit": [], "crossfit": []}

    for rep in range(n_reps):
        if rep % 100 == 0:
            print(f"[{learner_name}] Replication {rep}/{n_reps}...")
        res = run_single_rep(learner_class, n_obs, alpha, random_state=rep)
        if res is not None:
            results["nonorth"].append(res["nonorth"])
            results["nosplit"].append(res["nosplit"])
            results["crossfit"].append(res["crossfit"])

    print(f"[{learner_name}] Valid reps: {len(results['nonorth'])}/{n_reps}")
    return {k: np.array(v) for k, v in results.items()}


def plot_experiment_1(results: dict, learner_name: str,
                      save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = {
        "nonorth": "Non-orthogonal ML",
        "nosplit": "DML (no sample splitting)",
        "crossfit": "DML (with cross-fitting)"
    }
    colors = ["coral", "orange", "steelblue"]
    xx = np.linspace(-6, 6, 200)
    yy = stats.norm.pdf(xx)

    for ax, (key, label), color in zip(axes, labels.items(), colors):
        ax.hist(results[key], bins=40, density=True,
                color=color, alpha=0.6, label=label)
        ax.plot(xx, yy, 'k-', linewidth=1.5, label='N(0,1)')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim([-6, 6])
        ax.set_xlabel(r'$(\hat{\theta} - \theta_0) / \hat{\sigma}$')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.suptitle(f'Learner: {learner_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


LEARNERS = {
    "Lasso": LassoLearner,
    "ElasticNet": ElasticNetLearner,
    "RandomForest": RandomForestLearner,
    "NeuralNet": NeuralNetLearner,
    "CausalForest": CausalForestLearner,
}