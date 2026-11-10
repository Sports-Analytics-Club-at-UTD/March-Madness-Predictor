"""Men's model-only research: frozen pre-game regular-season snapshots.

Regular training games are days 101--132; their features stop at day 100.
Tournament features stop at day 132 (ordinals 133). No seeds in this model.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from . import features as F


@lru_cache(None)
def table(cutoff):
    games = F.team_games('M')
    games = games[games.DayNum <= cutoff].copy()
    games['poss'] = (F.possessions(games, 'T1_') + F.possessions(games, 'T2_')) / 2
    games['eff'] = 100 * games.T1_Score / games.poss
    out = []
    for year, s in games.groupby('Season'):
        teams = sorted(set(s.T1) | set(s.T2))
        idx = {t: i for i, t in enumerate(teams)}
        n, nt = len(s), len(teams)
        r = np.arange(n)
        a, b = s.T1.map(idx).values, s.T2.map(idx).values
        X = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                              (np.r_[r, r], np.r_[a, b])), shape=(n, nt))
        X = sparse.hstack([X, s['loc'].values[:, None]]).tocsr()
        z = pd.DataFrame({'Season': year, 'TeamID': teams})
        for rec in (0, 1.5):
            w = np.exp(rec * (s.DayNum.values - cutoff) / 132)
            m = Ridge(alpha=2, fit_intercept=False).fit(X, np.clip(s.margin, -40, 40), sample_weight=w)
            z['q' if rec == 0 else 'qrec'] = m.coef_[:nt]
        XO = sparse.csr_matrix((np.ones(2*n), (np.r_[r,r], np.r_[a,nt+b])), shape=(n,2*nt))
        XO = sparse.hstack([XO,s['loc'].values[:,None]]).tocsr()
        m = Ridge(alpha=10).fit(XO,s.eff)
        z['off'], z['def'] = m.coef_[:nt], -m.coef_[nt:2*nt]
        z['tempo'] = z.TeamID.map(s.groupby('T1').poss.mean())
        out.append(z)
    z = pd.concat(out, ignore_index=True)
    ranks = F.read('MMasseyOrdinals')
    ranks = ranks[ranks.RankingDayNum <= (133 if cutoff == 132 else cutoff)]
    last = ranks.sort_values('RankingDayNum').groupby(['Season','SystemName','TeamID']).tail(1)
    p = last.pivot(index=['Season','TeamID'],columns='SystemName',values='OrdinalRank')
    cols = ['POM','SAG','MOR','WLK','DOL','COL','MAS','RTH','WOL','BIH']
    p = p.reindex(columns=cols)
    p = -np.log(p)
    p['consensus'] = p.median(axis=1)
    p = p.T.fillna(p['consensus']).T
    return z.merge(p.reset_index(),on=['Season','TeamID'],how='left')


@lru_cache(None)
def training_rows(neutral):
    d = F.read('MRegularSeasonCompactResults')
    d = d[(d.DayNum > 100) & (d.DayNum <= 132)]
    if neutral:
        d = d[d.WLoc == 'N']
    a = pd.DataFrame({'Season':d.Season,'T1':d.WTeamID,'T2':d.LTeamID,
                      'margin':d.WScore-d.LScore,'loc':d.WLoc.map({'H':1,'A':-1,'N':0})})
    b = a.rename(columns={'T1':'T2','T2':'T1'}).copy()
    b[['margin','loc']] *= -1
    return pd.concat([a,b],ignore_index=True)


def design(pairs, cutoff, feats):
    return F.matchup_frame(pairs,table(cutoff),feats)[['diff_'+f for f in feats]].fillna(0).values


def predict(cfg, year):
    from . import backtest as B
    opts = cfg.get('mens_allyears',{})
    feats = opts.get('features',['q','qrec','off','def','consensus','POM'])
    tr = F.tourney_games('M')
    tr = tr[(tr.Season >= 2003) & (tr.Season < year)]
    X = design(tr,132,feats)
    X = np.c_[X,np.zeros(len(X))]
    y = (tr.margin.values > 0).astype(int)
    weights = np.ones(len(tr))
    reg_w = opts.get('regular_weight',0)
    if reg_w:
        reg = training_rows(opts.get('neutral',False))
        reg = reg[(reg.Season >= 2003) & (reg.Season < year)]
        xr = np.c_[design(reg,100,feats),reg['loc'].values]
        # Equal total regular/tournament weight per available season at weight=1.
        counts = tr.groupby('Season').size()
        rw = reg.Season.map(counts).fillna(126).values / reg.Season.map(reg.groupby('Season').size()).values
        X = np.r_[X,xr]
        y = np.r_[y,(reg.margin.values > 0).astype(int)]
        weights = np.r_[weights,reg_w*rw]
    sc = StandardScaler().fit(X,sample_weight=weights)
    m = LogisticRegression(C=opts.get('C',.1),fit_intercept=False,max_iter=2000).fit(sc.transform(X),y,sample_weight=weights)
    te = B.test_pairs('M',year)
    xx = np.c_[design(te,132,feats),np.zeros(len(te))]
    return expit(m.decision_function(sc.transform(xx)))


@lru_cache(None)
def margin_table():
    from .mens_features import extra_table
    return F.team_table('M',q_clip=60).merge(extra_table(),on=['Season','TeamID'],how='left')


def predict_margin(opts, year):
    """Ridge margins with season-held-out logistic calibration (past seasons only)."""
    from . import backtest as B
    feats=opts['features']
    tr=F.tourney_games('M')
    tr=tr[(tr.Season >= 2003) & (tr.Season < year)]
    cols=['diff_'+f for f in feats]
    X=F.matchup_frame(tr,margin_table(),feats)[cols].fillna(0).values
    sc=StandardScaler().fit(X)
    X=sc.transform(X)
    margins=tr.margin.values
    oof=np.zeros(len(tr))
    for s in tr.Season.unique():
        hold=tr.Season.values==s
        m=Ridge(alpha=opts.get('alpha',100),fit_intercept=False).fit(X[~hold],margins[~hold])
        oof[hold]=m.predict(X[hold])
    cal=LogisticRegression(C=100,fit_intercept=False).fit(oof[:,None],(margins>0).astype(int))
    m=Ridge(alpha=opts.get('alpha',100),fit_intercept=False).fit(X,margins)
    te=B.test_pairs('M',year)
    xx=F.matchup_frame(te,margin_table(),feats)[cols].fillna(0).values
    return cal.predict_proba(m.predict(sc.transform(xx))[:,None])[:,1]
