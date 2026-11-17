"""Issue #67 candidates. Reference is women_recommended.RECOMMENDED (men = final). Only the named part changes;
market weights, market sharpening and the women's 1.1 sharpening are inherited unchanged.

Run: uv run --extra training python -m kaggle_tuning.strat_reeval kaggle_tuning/configs/strat_reeval.py
"""
from kaggle_tuning.configs.final import X
from kaggle_tuning.configs.women_recommended import RECOMMENDED, WOMEN

REF = RECOMMENDED
POST = {k: v for k, v in REF.items() if k not in ("models", "by_gender")}
MEMBERS = {"M": REF["models"], "W": WOMEN["models"]}  # men: L 1, X 1; women: L(.1) .5, X .1, original .4


def combiner(model, **opts):
    return {**POST, "model": model, "members": MEMBERS, "combiner": opts}


def swap_tree(model):
    """Replace the xgb_spline member in both genders' blends (same features, params, weight)."""
    tree = {**X, "model": model}
    swap = lambda ms: [(tree if m is X else m, w) for m, w in ms]
    return {**REF, "models": swap(REF["models"]), "by_gender": {"W": {"models": swap(WOMEN["models"])}}}


CONFIGS = {
    "ref": REF,
    # A. Hedge, PR #66 protocol: uniform prior, eta = 8 per season-mean Brier, HIST seasons < Y only.
    "hedge": combiner("strat_hedge", uniform=True, eta=8.0),
    # Sensitivity: start Hedge from the current hand-set weights instead of uniform.
    "hedge_refprior": combiner("strat_hedge", eta=8.0),
    # B. Stacking: logistic regression on member log odds, history = 2012-2014 warm-up + HIST seasons < Y.
    "stack": combiner("strat_stack", lam=0.0, warmup=(2012, 2013, 2014)),
    # Same, ridge-penalised toward the current weights (lam = 63, about one season of games).
    "stack_ridge": combiner("strat_stack", lam=63.0, warmup=(2012, 2013, 2014)),
    # Stacking learns its own temperature, so also drop the women's post-market 1.1 sharpening.
    "stack_nosharpen": {**combiner("strat_stack", lam=0.0, warmup=(2012, 2013, 2014)), "sharpen": {}},
    # C. GBM margin -> probability: normal-CDF map instead of the spline, or a logistic objective.
    "xgb_normal": swap_tree("xgb_normal"),
    "xgb_binary": swap_tree("xgb_binary"),
}
