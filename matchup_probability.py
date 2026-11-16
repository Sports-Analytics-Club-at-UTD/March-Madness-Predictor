"""Shared logistic matchup model used by simulation and historical evaluation."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def fit_symmetric_matchup(features, outcomes):
    """Fit the existing zero-intercept, C=1 model with both game perspectives."""
    features = np.asarray(features, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    return LogisticRegression(fit_intercept=False, C=1.0).fit(
        np.concatenate([features, -features]), np.concatenate([outcomes, 1 - outcomes])
    )


def matchup_probability(score_difference, seed_difference, model=None):
    """P(A beats B); seed_difference = seed[B] - seed[A]."""
    weights = (0.85, 0.12) if model is None else model.coef_[0]
    return expit(weights[0] * score_difference + weights[1] * seed_difference)
