"""Logistic models fitted exactly, by Newton's method, with nothing random in them.

RULES.md asks that running the repository reproduces the submission: the same file, not
approximately. An exact solver with no seed, no minibatches and no threads to race has no source of
run-to-run difference, and the fitted coefficients are small enough to read in a text file.
"""

import json

import numpy as np

# Params
L2 = 1e-3
NEWTON_STEPS = 20
TOLERANCE = 1e-6
# Coefficients are stored to this many decimals, so a rounding difference in the last bits of a
# fitted value on another machine cannot reach the submission.
COEF_DECIMALS = 6


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(X, y, intercept, l2=L2, weights=None):
    """dict(coef, intercept, mean, scale). `intercept` False leaves the columns uncentred.

    Antisymmetric columns have to stay uncentred and intercept-free: that is what keeps the fitted
    model antisymmetric, so it has no way of preferring one team.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    mean = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    Z = (X - mean) / scale
    if intercept:
        Z = np.column_stack([Z, np.ones(len(Z))])

    penalty = np.full(Z.shape[1], l2 * weights.sum())
    if intercept:
        penalty[-1] = 0.0
    beta = np.zeros(Z.shape[1])
    for _ in range(NEWTON_STEPS):
        p = _sigmoid(Z.dot(beta))
        curvature = np.maximum(p * (1.0 - p), 1e-6) * weights
        gradient = Z.T.dot((p - y) * weights) + penalty * beta
        hessian = (Z.T * curvature).dot(Z) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < TOLERANCE:
            break
    return {
        "coef": np.round(beta[:X.shape[1]], COEF_DECIMALS).tolist(),
        "intercept": float(np.round(beta[-1], COEF_DECIMALS)) if intercept else 0.0,
        "mean": np.round(mean, COEF_DECIMALS).tolist(),
        "scale": np.round(scale, COEF_DECIMALS).tolist(),
    }


def predict_logit(model, X):
    """Log-odds, one per row."""
    coef = np.array(model["coef"], dtype=np.float64)
    mean = np.array(model["mean"], dtype=np.float64)
    scale = np.array(model["scale"], dtype=np.float64)
    return ((np.asarray(X, dtype=np.float64) - mean) / scale).dot(coef) + model["intercept"]


def save_models(path, models):
    with open(path, "w") as handle:
        json.dump(models, handle, indent=1, sort_keys=True)
        handle.write("\n")


def load_models(path):
    with open(path) as handle:
        return json.load(handle)
