"""Current best (2025 #90, 2026 #56). Subagents: import BASE and extend it."""
BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
Q = {"q_clip_M": 60, "q_clip_W": 60}
L = {"M": M0, "W": W0, "model": "logit", "C": 0.1, **Q}
X = {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **Q, "xgb": {"max_depth": 2}}
ENS = {"models": [(L, 1), (X, 1)], **Q}
BASE = {**ENS, "market_w": {"M": 0.85, "W": 0.5}, "mkt_sharpen": 1.2}
CONFIGS = {"final_model_only": ENS, "final": BASE}
