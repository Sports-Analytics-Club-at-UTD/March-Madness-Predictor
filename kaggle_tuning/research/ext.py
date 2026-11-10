"""External pre-tournament rating/market sources as pairwise P(T1 beats T2) for T1<T2 Kaggle IDs.

 - fte(year,g):  FiveThirtyEight Selection-Sunday power ratings (2016-2019, 2021-2023; dataset kaito510/538data,
                 published 2025-03-18 but each snapshot is dated forecast_date = the Selection Sunday of that season).
                 P = 1/(1+10^(-(r1-r2)*30.464/400)).
 - gelo(year,g): goto_conversion Elo table (rd1..rd6 + Elo) -> logistic of Elo diff (unclipped). 2024 + 2025.
 - gsub(year,g): goto_conversion submission (2025, 2026) as used in final.py (clipped to ~[.025,.975] per round).
"""
from functools import lru_cache
import numpy as np, pandas as pd
from .gtab import EXT, MIR, table, ids, seeds

FTE = {2016: 'fivethirtyeight_ncaa_forecasts (6).csv', 2017: 'fivethirtyeight_ncaa_forecasts.csv', 2018: 'fivethirtyeight_ncaa_forecasts (1).csv',
       2019: 'fivethirtyeight_ncaa_forecasts (2).csv', 2021: 'fivethirtyeight_ncaa_forecasts (3).csv', 2022: 'fivethirtyeight_ncaa_forecasts (4).csv',
       2023: 'fivethirtyeight_ncaa_forecasts (5).csv'}
nz = lambda x: ''.join(c for c in str(x).lower() if c.isalnum())


def _allpairs(rows):
    """rows: DataFrame[TeamID, r] -> all T1<T2 pairs with rating diff"""
    a = rows.rename(columns={'TeamID': 'T1', 'r': 'r1'}); b = rows.rename(columns={'TeamID': 'T2', 'r': 'r2'})
    m = a.merge(b, how='cross'); m = m[m.T1 < m.T2]
    return m


@lru_cache(maxsize=None)
def fte(year, g):
    d = pd.read_csv(EXT / 'market25' / FTE[year]); d = d[d.gender == {'M': 'mens', 'W': 'womens'}[g]]
    d = d[d.forecast_date == d.forecast_date.min()]
    sp = pd.read_csv(MIR / f'{g}TeamSpellings.csv', encoding='cp1252'); m = {nz(r.TeamNameSpelling): r.TeamID for r in sp.itertuples()}
    tn = pd.read_csv(MIR / f'{g}Teams.csv'); m2 = {nz(r.TeamName): r.TeamID for r in tn.itertuples()}
    rows = []
    for r in d.itertuples():
        k = nz(r.team_name); tid = m.get(k, m2.get(k))
        if tid is None: tid = m.get(nz(r.team_name.replace('St.', 'State')), None)
        rows.append((tid, r.team_rating, r.team_name))
    df = pd.DataFrame(rows, columns=['TeamID', 'r', 'name'])
    bad = df[df.TeamID.isna()]
    if len(bad): print('UNMATCHED', year, g, bad.name.tolist())
    df = df.dropna(subset=['TeamID']).astype({'TeamID': int})
    pr = _allpairs(df[['TeamID', 'r']])
    return pd.DataFrame({'T1': pr.T1.values, 'T2': pr.T2.values, 'lo': (pr.r1 - pr.r2).values * 30.464 / 400 * np.log(10)})


@lru_cache(maxsize=None)
def gelo(year, g):
    t = table(year, g); I = ids(year, g, t); sd = seeds(year, g)
    rows = []
    for (p, cand, e) in zip(t.player, I, t.Elo_Rating):
        for c in cand: rows.append((c, e))   # a First Four placeholder's strength applies to both candidates
    df = pd.DataFrame(rows, columns=['TeamID', 'r'])
    pr = _allpairs(df)
    return pd.DataFrame({'T1': pr.T1.values, 'T2': pr.T2.values, 'lo': (pr.r1 - pr.r2).values / 400 * np.log(10)})


@lru_cache(maxsize=None)
def gsub(year, g):
    p = EXT / ('market25' if year == 2025 else 'market26') / 'submission.csv'
    d = pd.read_csv(p); T1 = d.ID.str[5:9].astype(int); T2 = d.ID.str[10:14].astype(int)
    lo = 1000 if g == 'M' else 3000
    m = (T1 >= lo) & (T1 < lo + 1000)
    q = np.clip(d.Pred[m].values, 1e-4, 1 - 1e-4)
    return pd.DataFrame({'T1': T1[m].values, 'T2': T2[m].values, 'lo': np.log(q / (1 - q))})


KR = {}  # round slope sub/elo estimated from 2025, per gender (filled lazily)


def round_slopes(g, year=2025):
    from .gtab import seeds, meet_round
    a = gsub(year, g); b = gelo(year, g); m = a.merge(b, on=['T1', 'T2'], suffixes=('_sub', '_elo'))
    sd = seeds(year, g); f = meet_round(year, g)
    m['rnd'] = [f(sd[x][:3], sd[y][:3]) for x, y in zip(m.T1, m.T2)]
    return {r: float((d.lo_sub * d.lo_elo).sum() / (d.lo_elo ** 2).sum()) for r, d in m.groupby('rnd')}


@lru_cache(maxsize=None)
def gdecl(year, g, ref=2025):
    """De-clipped reconstruction of the goto_conversion pair log-odds from the *submission only*:
    solve lo_sub(i,j) = k_r(i,j) * (e_i - e_j) for per-team strengths e (k_r = round slopes measured on the 2025
    sub-vs-Elo comparison, same gender). Returns pairwise logit(e_i - e_j)  (an Elo-scale, unshrunk rating)."""
    from .gtab import seeds, meet_round
    k = round_slopes(g, ref)
    a = gsub(year, g); sd = seeds(year, g); f = meet_round(year, g)
    a = a[a.T1.isin(sd) & a.T2.isin(sd)].copy()
    a['rnd'] = [f(sd[x][:3], sd[y][:3]) for x, y in zip(a.T1, a.T2)]
    a = a[a.rnd <= 6]
    teams = sorted(set(a.T1) | set(a.T2)); ix = {t: i for i, t in enumerate(teams)}
    A = np.zeros((len(a) + 1, len(teams))); y = np.zeros(len(a) + 1)
    for n, r in enumerate(a.itertuples()):
        kk = k[r.rnd]; A[n, ix[r.T1]] = kk; A[n, ix[r.T2]] = -kk; y[n] = r.lo
    A[-1, :] = 1e-3  # anchor
    e = np.linalg.lstsq(A, y, rcond=None)[0]
    pr = pd.DataFrame([(t1, t2) for t1 in teams for t2 in teams if t1 < t2], columns=['T1', 'T2'])
    pr['lo'] = [e[ix[x]] - e[ix[z]] for x, z in zip(pr.T1, pr.T2)]
    return pr


@lru_cache(maxsize=None)
def gshr(year, g, ref=2025):
    """Raw Elo table pair log-odds shrunk to mimic the goto_conversion *submission* format (per-round slopes k_r measured
    on the 2025 table-vs-submission comparison). Lets 2024 (table only) be scored with the same de-clip factor as 2025/26."""
    from .gtab import seeds, meet_round
    k = round_slopes(g, ref); e = gelo(year, g); sd = seeds(year, g); f = meet_round(year, g)
    rnd = np.array([f(sd[a][:3], sd[b][:3]) if a in sd and b in sd else 9 for a, b in zip(e.T1, e.T2)])
    out = e.copy(); out['lo'] = e.lo.values * np.array([k.get(r, 1.0) for r in rnd]); return out
