BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **k}
MK = {"M": .5, "W": .5}
CONFIGS = {
  "logit_CW1": L(C_W=1.0),
  "logit_CW10": L(C_W=10.0),
  "logit_C1": L(C=1.0),
  "logit_qclipW60": L(q_clip_W=60),
  "logit_qrec": L(M=M0+["quality_rec"], W=W0+["quality_rec"]),
  "logit_mworst": L(M=M0+["mstr_worst","mstr_mean3"]),
  "logit_small": L(M=["seed","quality","mstr_mean3"], W=["seed","quality"]),
  "ens_lx": {"models": [(L(), 1), (X(), 1)]},
  "ens_lx_mk.5": {"models": [(L(), 1), (X(), 1)], "market_w": MK},
  "ens_lx_CW10_mk.5": {"models": [(L(C_W=10.0), 1), (X(), 1)], "market_w": MK},
  "logit_CW10_mk.5": L(C_W=10.0, market_w=MK),
}
