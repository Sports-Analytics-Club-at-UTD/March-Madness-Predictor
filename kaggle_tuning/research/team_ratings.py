"""Regular-season-only adjusted ratings; tuning sees strictly earlier tournaments."""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from scipy.special import expit
from kaggle_tuning import features as F

ALPHAS = (.1, 1., 10., 100.)


def margin_design(s, teams):
    idx = {t:i for i,t in enumerate(teams)}
    n = len(s)
    a, b, r = s.T1.map(idx).to_numpy(), s.T2.map(idx).to_numpy(), np.arange(n)
    X = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                          (np.r_[r,r], np.r_[a,b])), shape=(n,len(teams)))
    return sparse.hstack([X, sparse.csr_matrix(s['loc'].to_numpy()[:,None])]).tocsr(), a, b


def gcv_alpha(X, y):
    gram = (X.T @ X).toarray()
    vals, vec = np.linalg.eigh(gram)
    z = vec.T @ (X.T @ y)
    def loss(alpha):
        df = np.sum(vals/(vals+alpha))
        rss = y @ y - np.sum(z*z*(vals+2*alpha)/(vals+alpha)**2)
        return rss / (len(y)-df)**2
    return min(ALPHAS, key=loss)


@lru_cache(maxsize=2)
def rating_table(g):
    d = F.team_games(g).query('DayNum <= 132').copy()
    out, previous = [], {}
    tournaments = F.tourney_games(g).query('T1 < T2')
    history = {a:[] for a in ALPHAS}
    for year, both in d.groupby('Season', sort=True):
        s = both.query('T1 < T2')
        teams = np.unique(np.r_[s.T1, s.T2]); nt = len(teams)
        X, a, b = margin_design(s, teams)
        y = np.clip(s.margin.to_numpy(dtype=float), -60, 60)
        z = pd.DataFrame({'Season':year,'TeamID':teams})
        fits = {alpha:Ridge(alpha=alpha,fit_intercept=False,solver='lsqr',tol=1e-8).fit(X,y).coef_ for alpha in ALPHAS}
        # Select alpha using earlier tournament Brier with a fixed ten-point logistic scale.
        selected = min(ALPHAS, key=lambda alpha: np.mean(history[alpha]) if history[alpha] else abs(np.log10(alpha)))
        z['rt_wf'] = fits[selected][:nt]
        z['rt_gcv'] = fits[gcv_alpha(X,y)][:nt]
        for alpha in ALPHAS:
            z[f'rt_margin{alpha:g}'] = fits[alpha][:nt]
        for strength in (1.5, 3.):
            w = np.exp(strength*(s.DayNum.to_numpy()-132)/132)
            z[f'rt_rec{strength:g}'] = Ridge(alpha=1,fit_intercept=False).fit(X,y,sample_weight=w).coef_[:nt]
        q = fits[1.][:nt]
        # High-quality matchups get at most twice the weight of the weakest matchups.
        percentile = pd.Series(q).rank(pct=True).to_numpy()
        w = 1 + (percentile[a]+percentile[b])/2
        z['rt_opp'] = Ridge(alpha=1,fit_intercept=False).fit(X,y,sample_weight=w).coef_[:nt]
        for alpha in (.01, .1):
            z[f'rt_lasso{alpha:g}'] = Lasso(alpha=alpha,fit_intercept=False,max_iter=5000,tol=1e-6).fit(X,y).coef_[:nt]
        coef = fits[1.].copy()
        for _ in range(12):
            residual = s.margin.to_numpy()-X@coef
            weights = np.minimum(1,10/np.maximum(abs(residual),1e-9))
            coef = Ridge(alpha=1,fit_intercept=False).fit(X,s.margin.to_numpy(),sample_weight=weights).coef_
        z['rt_huber'] = coef[:nt]
        for strength in (1.,10.):
            prior = np.r_[[.7*previous.get(int(t),0) for t in teams],0.]
            coef = Ridge(alpha=strength,fit_intercept=False).fit(X,y-X@prior).coef_+prior
            z[f'rt_carry{strength:g}'] = coef[:nt]
        previous = dict(zip(teams,q))
        z['rt_bt'] = LogisticRegression(C=1,fit_intercept=False,max_iter=1000).fit(X,(s.margin>0).to_numpy()).coef_[0,:nt]
        # Two observations per game, jointly estimate offense and opponent defense.
        XO, aa, bb = margin_design(both,teams)
        n = len(both); r = np.arange(n)
        XO = sparse.csr_matrix((np.ones(2*n),(np.r_[r,r],np.r_[aa,nt+bb])),shape=(n,2*nt))
        XO = sparse.hstack([XO,sparse.csr_matrix(both['loc'].to_numpy()[:,None])]).tocsr()
        poss = (F.possessions(both,'T1_')+F.possessions(both,'T2_')).to_numpy()/2
        target = 100*both.T1_Score.to_numpy()/poss
        target -= target.mean()
        for alpha in (1.,10.,100.):
            coef = Ridge(alpha=alpha,fit_intercept=False).fit(XO,target).coef_
            z[f'rt_off{alpha:g}'], z[f'rt_def{alpha:g}'] = coef[:nt], -coef[nt:2*nt]
            z[f'rt_net{alpha:g}'] = coef[:nt]-coef[nt:2*nt]
        for col in z.columns[2:]:
            z[col] -= z[col].mean()
        out.append(z)
        # Record this season only AFTER choosing its alpha; cannot affect its own features.
        games = tournaments.query('Season == @year')
        idx = {t:i for i,t in enumerate(teams)}
        for alpha,coef in fits.items():
            delta = games.T1.map(idx).map(lambda i:coef[int(i)] if pd.notna(i) else 0).to_numpy()-games.T2.map(idx).map(lambda i:coef[int(i)] if pd.notna(i) else 0).to_numpy()
            if len(games):
                history[alpha].append(float(np.mean((expit(delta/10)-(games.margin.to_numpy()>0))**2)))
    return pd.concat(out,ignore_index=True)
