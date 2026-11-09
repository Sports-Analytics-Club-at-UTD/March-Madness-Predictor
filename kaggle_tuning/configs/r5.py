BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
Q = {"q_clip_M": 60, "q_clip_W": 60}
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **Q, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **Q, "xgb": {"max_depth": 2}, **k}
CONFIGS = {
  "L_q60": L(),
  "L_q60_Wbox": L(W=W0+BOXF),
  "L_q60_qa.1_Wbox": L(W=W0+BOXF, q_alpha=0.1),
  "L_q60_noclip": L(q_clip_M=1000, q_clip_W=1000),
  "L_q60_Wbox_CW.3": L(W=W0+BOXF, C_W=0.3),
  "L_q60_Mworst": L(M=M0+["mstr_worst","mstr_mean3"]),
  "L_q60_Mbox4": L(M=M0+["efg","opp_efg","tov","orb"]),
  "X_d2_q60": X(),
  "X_d2_q60_d1": X(xgb={"max_depth": 1}, n_est=1000),
  "E_q60": {"models": [(L(W=W0+BOXF), 1), (X(), 1)], **Q},
  "E_q60_2to1": {"models": [(L(W=W0+BOXF), 2), (X(), 1)], **Q},
}
