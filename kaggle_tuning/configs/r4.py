BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **k}
CONFIGS = {
  "L": L(),
  "L_qa0.1": L(q_alpha=0.1), "L_qa5": L(q_alpha=5.0), "L_qa20": L(q_alpha=20.0),
  "L_qclip15": L(q_clip_M=15, q_clip_W=15),
  "L_qclip60": L(q_clip_M=60, q_clip_W=60),
  "L_C0.03": L(C=0.03), "L_C0.3": L(C=0.3),
  "L_box": L(M=M0+BOXF, W=W0+BOXF),
  "L_start2008": L(start_M=2008, start_W=2012),
  "X_d2": X(xgb={"max_depth": 2}), "X_mcw10": X(xgb={"min_child_weight": 10}),
  "X_small": X(M=M0, W=W0),
  "X_lr01_1000": X(xgb={"learning_rate": 0.01}, n_est=1000),
}
