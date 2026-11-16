"""SHAP dot-product normalization shared by the scorer and historical exporter."""

import numpy as np


def dot_zscore(features, scaled_features, shap_weights):
    used = [feature for feature in shap_weights if feature in features.columns]
    if not used:
        return np.zeros(len(features), dtype=float), list(shap_weights)
    missing = [feature for feature in shap_weights if feature not in features.columns]
    weights = np.array([shap_weights[feature] for feature in used], dtype=float)
    raw = scaled_features[:, [features.columns.get_loc(feature) for feature in used]] @ weights
    std = raw.std() if raw.std() > 0 else 1.0
    return (raw - raw.mean()) / std, missing
