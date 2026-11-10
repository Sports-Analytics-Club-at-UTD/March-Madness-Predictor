"""Load goto_conversion advancement tables (rd1..rd6 + Elo) and map rows to Kaggle TeamIDs.

Table rows are in bracket order (pairs of consecutive rows are the R1 games). First Four
slots appear as placeholder names like 'Y16'/'W11' (seed slot); they resolve to whichever
team ended up holding that seed (placeholder strength applies to the actual First Four winner)."""
import numpy as np, pandas as pd
from pathlib import Path
EXT = Path('/Users/spro/code/March-Madness-Predictor/data/kaggle')
PK = EXT / 'market26/pkg/goto_conversion/outputFiles'
MIR = EXT / 'mirror26'

def table(year, g):
    g_ = {'M': 'mens', 'W': 'womens'}[g]
    if year == 2025 and False:
        return pd.read_csv(EXT / f'market25/{g_}ProbabilitiesTable.csv')
    return pd.read_csv(PK / f'{g_}ProbabilitiesTable{year}.csv')

def ids(year, g, t):
    """returns TeamID (or list of candidate ids for placeholders) per row"""
    pre = g
    sp = pd.read_csv(MIR / f'{pre}TeamSpellings.csv', encoding='cp1252')
    nz = lambda x: ''.join(c for c in str(x).lower() if c.isalnum())
    m = {nz(r.TeamNameSpelling): r.TeamID for r in sp.itertuples()}
    tn = pd.read_csv(MIR / f'{pre}Teams.csv'); nm = {nz(r.TeamName): r.TeamID for r in tn.itertuples()}
    sd = pd.read_csv(MIR / f'{pre}NCAATourneySeeds.csv'); sd = sd[sd.Season == year]
    out = []
    for p in t.player:
        k = nz(p)
        if k in m: out.append([m[k]])
        elif k in nm: out.append([nm[k]])
        else: out.append(sd[sd.Seed == p].TeamID.tolist() or sd[sd.Seed.str[:3] == p].TeamID.tolist())
    return out


def seeds(year, g):
    sd = pd.read_csv(MIR / f'{g}NCAATourneySeeds.csv'); sd = sd[sd.Season == year]
    return {r.TeamID: r.Seed for r in sd.itertuples()}

def slots(year, g):
    s = pd.read_csv(MIR / f'{g}NCAATourneySlots.csv'); return s[s.Season == year]

def meet_round(year, g):
    """returns f(seedA, seedB) -> round (1..6) of potential meeting, using 3-char seed codes."""
    s = slots(year, g)
    ch = {r.Slot: (r.StrongSeed, r.WeakSeed) for r in s.itertuples() if r.Slot.startswith('R')}
    def leaves(x):
        if x in ch: return leaves(ch[x][0]) | leaves(ch[x][1])
        return {x[:3]}
    rnd = {sl: int(sl[1]) for sl in ch}
    L = {sl: leaves(sl) for sl in ch}
    def f(a, b):
        best = 9
        for sl, lv in L.items():
            if a in lv and b in lv: best = min(best, rnd[sl])
        return best
    return f

def rowmap(year, g, t):
    """row index -> 3-char seed code"""
    sd = seeds(year, g); I = ids(year, g, t); out = []
    for p, cand in zip(t.player, I):
        if p in set(v[:3] for v in sd.values()): out.append(p); continue
        c = [sd[x][:3] for x in cand if x in sd]
        out.append(c[0] if c else None)
    return out


RD = ['rd1_win', 'rd2_win', 'rd3_win', 'rd4_win', 'rd5_win', 'rd6_win']

def goto(odds, total=1.0, eps=1e-6):
    p = [1 / x for x in odds]; se = [((x - x * x) / x) ** .5 for x in p]
    step = (sum(p) - total) / sum(se)
    return [min(max(x - y * step, eps), 1) for x, y in zip(p, se)]

def goto_pre(odds):  # kaito510 notebook version incl. pre-processing when sum(prob) < 1
    p = [1 / x for x in odds]
    if sum(p) < 1:
        mx = .99 / max(p); p = [x * mx for x in p]
        r = goto([1 / (1 - x) for x in p]); return [1 - x for x in r]
    return goto(odds)

def pairs_from_table(year, g, method='goto_pre'):
    """dict (seedA,seedB)->P(A beats B) for all potential meetings, from advancement probs."""
    t = table(year, g); rows = rowmap(year, g, t); f = meet_round(year, g)
    R = t[RD].values; n = len(t)
    elo = t.Elo_Rating.values
    out = {}
    for i in range(n):
        for j in range(n):
            if i == j: continue
            k = f(rows[i], rows[j])
            if k > 6: continue
            def w(x): return R[x, k - 1] / (R[x, k - 2] if k > 1 else 1.0)
            a, b = w(i), w(j)
            if method == 'elo':
                pr = 1 / (1 + 10 ** (-(elo[i] - elo[j]) / 400))
            elif method == 'mult':
                pr = a / (a + b)
            elif method == 'goto_pre':
                pr = goto_pre([1 / a, 1 / b])[0]
            out[(rows[i], rows[j])] = (pr, k)
    return out
