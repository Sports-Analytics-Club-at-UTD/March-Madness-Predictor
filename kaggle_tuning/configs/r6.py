BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
Q = {"q_clip_M": 60, "q_clip_W": 60}
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **Q, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **Q, "xgb": {"max_depth": 2}, **k}
CONFIGS = {}
for tag, lw in [("Wbox", L(W=W0+BOXF)), ("W0", L())]:
    for wm, ww in [(0.85, 0.5), (0.7, 0.5), (0.85, 0.3), (0.7, 0.4)]:
        CONFIGS[f"E_{tag}_mkM{wm}_W{ww}"] = {"models": [(lw, 1), (X(), 1)], **Q, "market_w": {"M": wm, "W": ww}, "mkt_sharpen": 1.2}
