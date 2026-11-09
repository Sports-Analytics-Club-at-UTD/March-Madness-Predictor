BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **k}
E = lambda **k: {"models": [(L(), 1), (X(), 1)], **k}
CONFIGS = {}
for wm in [0.5, 0.7, 0.85]:
    for ww in [0.3, 0.5]:
        for ms in [1.0, 1.2]:
            CONFIGS[f"ens_mkM{wm}_W{ww}_ms{ms}"] = E(market_w={"M": wm, "W": ww}, mkt_sharpen=ms)
CONFIGS["mkt_only_ms1.2"] = L(market_w={"M": 1.0, "W": 1.0}, mkt_sharpen=1.2)
CONFIGS["mkt_only"] = L(market_w={"M": 1.0, "W": 1.0})
