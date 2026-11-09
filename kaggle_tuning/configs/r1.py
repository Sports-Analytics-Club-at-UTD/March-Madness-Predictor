BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
L = lambda **k: {"M": M0, "W": W0, "model": "logit", "C": 0.1, **k}
X = lambda **k: {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **k}
CONFIGS = {
  "logit": L(),
  "logit_shW1.3": L(sharpen={"W": 1.3}),
  "logit_mk.5": L(market_w={"M": .5, "W": .5}),
  "logit_mk.5_shW1.3": L(market_w={"M": .5, "W": .5}, sharpen={"W": 1.3}),
  "logit_mk.7": L(market_w={"M": .7, "W": .7}),
  "logit_mk.3": L(market_w={"M": .3, "W": .3}),
  "xgb": X(),
  "xgb_mk.5": X(market_w={"M": .5, "W": .5}),
  "xgb_mk.3": X(market_w={"M": .3, "W": .3}),
  "xgb_mkM.5": X(market_w={"M": .5}),
  "logit_mkM.6_W.3": L(market_w={"M": .6, "W": .3}),
}
