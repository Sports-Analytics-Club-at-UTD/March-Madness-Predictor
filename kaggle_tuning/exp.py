"""Ad-hoc experiment runner: python -m kaggle_tuning.exp"""
import sys, json
from .backtest import run
BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
EXPS = {
 "q_only": {"M":["quality"],"W":["quality"],"model":"logit"},
 "q_seed": {"M":["quality","seed"],"W":["quality","seed"],"model":"logit"},
 "q_massey": {"M":["quality","mstr_med","seed"],"W":["quality","seed"],"model":"logit"},
 "base_C01": {"M":M0,"W":W0,"model":"logit","C":0.1},
 "xgb_raddar": {"M":["seed","quality"]+BOXF,"W":["seed","quality"]+BOXF,"model":"xgb_spline"},
 "xgb_base": {"M":M0+BOXF,"W":W0+BOXF,"model":"xgb_spline"},
}
names = sys.argv[1:] or list(EXPS)
for n in names:
    print("==", n); run(EXPS[n])
