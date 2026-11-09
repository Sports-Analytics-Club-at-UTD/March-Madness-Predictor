"""Men-only ablations; the women's model and market settings stay fixed."""

from kaggle_tuning.configs.final import BASE, BOXF, M0, L, X


def variant(replace=None, add=(), model="ensemble", **kwargs):
    replace = replace or {}
    feats = [replace.get(f, f) for f in M0] + list(add)
    l = {**L, "M": feats, "mens_features": True, **kwargs}
    x = {**X, "M": feats + BOXF, "mens_features": True, **kwargs}
    return {
        **BASE,
        "men_models": [(l, 1), (x, 1)]
        if model == "ensemble"
        else [(l if model == "logit" else x, 1)],
    }


CONFIGS = {"baseline": BASE}
for name, replacement in [
    ("pool", "q_pool"),
    ("dayseed", "q_dayseed"),
    ("ft", "q_ft"),
    ("raddar", "q_raddar"),
]:
    CONFIGS[name] = variant({"quality": replacement})
CONFIGS["elo5"] = variant({"elo": "elo5"})
CONFIGS["conference"] = variant(add=["conf_quality"])
for alpha in [10, 100]:
    CONFIGS[f"offdef{alpha}"] = variant(
        {"net_eff": f"adj_net{alpha}"}, add=[f"adj_o{alpha}", f"adj_d{alpha}"]
    )
CONFIGS["worst"] = variant({"mstr_POM": "mstr_worst"})
CONFIGS["normal"] = variant({"mstr_POM": "normal_POM", "mstr_med": "normal_consensus"})
CONFIGS["normal_worst"] = variant(
    {"mstr_POM": "normal_worst", "mstr_med": "normal_consensus"}
)
for agg in ["mean", "median", "min"]:
    CONFIGS[f"recent_{agg}"] = variant(
        {"mstr_med": f"recent_{agg}", "mstr_POM": f"recent_POM_{agg}"}
    )
CONFIGS["more_systems"] = variant(
    add=[
        "log_SAG",
        "log_MOR",
        "log_WLK",
        "log_DOL",
        "log_COL",
        "log_MAS",
        "log_RTH",
        "log_WOL",
        "log_BIH",
    ]
)
CONFIGS["learned_consensus"] = variant(
    add=[
        "normal_SAG",
        "normal_MOR",
        "normal_WLK",
        "normal_DOL",
        "normal_COL",
        "normal_MAS",
        "normal_RTH",
        "normal_WOL",
        "normal_BIH",
    ],
    C=0.01,
)
CONFIGS["form"] = variant(add=["late_adj", "form_delta", "late_stability"])
for c in [0.01, 0.1, 1.0]:
    CONFIGS[f"logit_C{c}"] = variant(model="logit", C=c)
for depth in [1, 3]:
    CONFIGS[f"depth{depth}"] = variant(xgb={"max_depth": depth, "n_jobs": 1})
CONFIGS["regularized"] = variant(
    xgb={"max_depth": 2, "min_child_weight": 100, "reg_lambda": 10, "n_jobs": 1}
)
CONFIGS["quality_recency"] = variant({"quality": "quality_rec"})
CONFIGS["raddar_bundle"] = variant(
    {"quality": "q_raddar", "elo": "elo5", "mstr_POM": "mstr_worst"},
    add=["conf_quality"],
)
# Combination candidates chosen for their separate historical gains.
CONFIGS["pool_worst"] = variant({"quality": "q_pool", "mstr_POM": "mstr_worst"})
CONFIGS["pool_worst_conf"] = variant(
    {"quality": "q_pool", "mstr_POM": "mstr_worst"}, add=["conf_quality"]
)
CONFIGS["pool_worst_form"] = variant(
    {"quality": "q_pool", "mstr_POM": "mstr_worst"},
    add=["late_adj", "form_delta", "late_stability"],
)
