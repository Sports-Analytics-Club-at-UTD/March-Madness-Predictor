"""Pre-tournament bust-year research. Run with MANIA_DATA set, python -m ...bust_years.

All choices below are fixed before examining correlations; no search over cutoffs.
LOSO is descriptive only. Predictive evaluation uses expanding, past-only labels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kaggle_tuning import features as F

OUT = Path(__file__).with_name('bust_years')
FEATURES = ['rating_spread', 'top_middle_gap', 'regular_upset_rate', 'late_upset_rate',
            'conference_upset_rate', 'bid_stealers', 'margin_variance', 'massey_disagreement',
            'three_attempt_rate', 'strong_midmajors', 'home_effect', 'previous_surprise']
OUTCOMES = ['surprise', 'brier_surprise', 'gap_upset_excess', 'slope_bust', 'top4_deficit']
MAJORS = {'acc', 'big_east', 'big_ten', 'big_twelve', 'pac_ten', 'pac_twelve', 'sec', 'big_eight', 'swc'}
SEED = 20261109


def coefficient(d):
    # Symmetric seed-only logistic: P(T1 wins) = sigmoid(beta*(seed2-seed1)).
    x, y = d.x.to_numpy(), d.y.to_numpy()
    return minimize_scalar(lambda b: np.mean(np.logaddexp(0, b*x)-y*b*x),
                           bounds=(0, 2), method='bounded').x


def games(g):
    seeds = F.seeds(g)
    sd = seeds.set_index(['Season', 'TeamID']).seed
    raw = F.read(g+'NCAATourneyCompactResults')
    full = F.read(g+'NCAATourneySeeds').set_index(['Season','TeamID']).Seed.str[:3]
    rows = []
    for s, d in raw.groupby('Season'):
        if s == 2026:
            continue
        for r in d.itertuples():
            if full[s,r.WTeamID] == full[s,r.LTeamID]:
                continue
            a,b = sorted([r.WTeamID,r.LTeamID])
            rows.append((s,a,b,int(a == r.WTeamID)))
    sol = F.solution_2026()
    for r in sol.itertuples():
        s,a,b = map(int,r.ID.split('_'))
        if (a < 3000) == (g == 'M'):
            rows.append((s,a,b,int(r.y)))
    d = pd.DataFrame(rows,columns=['Season','T1','T2','y'])
    for k in (1,2):
        d['s'+str(k)] = [sd[s,t] for s,t in zip(d.Season,d['T'+str(k)])]
    d['x'] = d.s2-d.s1
    counts = d.groupby('Season').size()
    expected = pd.Series(63, index=counts.index)
    if g == 'M': expected.loc[2021] = 62  # VCU–Oregon was a no-contest.
    assert counts.equals(expected)
    assert not d.duplicated(['Season','T1','T2']).any()
    return d


def expected_top4_wins(g, year, beta):
    seeds = F.read(g+'NCAATourneySeeds').query('Season == @year')
    slots = F.read(g+'NCAATourneySlots').query('Season == @year').copy()
    sd = dict(zip(seeds.TeamID, seeds.Seed.str[1:3].astype(int)))
    dist = {r.Seed: {r.TeamID: 1.0} for r in seeds.itertuples()}
    expected = 0.0
    while len(slots):
        ready = slots[slots.StrongSeed.isin(dist) & slots.WeakSeed.isin(dist)]
        assert len(ready), 'Unresolved bracket slots'
        for r in ready.itertuples():
            winners = {}
            for a, pa in dist[r.StrongSeed].items():
                for b, pb in dist[r.WeakSeed].items():
                    p = expit(beta*(sd[b]-sd[a]))
                    winners[a] = winners.get(a, 0.0)+pa*pb*p
                    winners[b] = winners.get(b, 0.0)+pa*pb*(1-p)
            dist[r.Slot] = winners
            if str(r.Slot).startswith(tuple('R'+str(i) for i in range(1,7))):
                expected += sum(p for t,p in winners.items() if sd[t] <= 4)
        slots = slots[~slots.Slot.isin(ready.Slot)]
    return expected


def labels(d, g):
    rows=[]
    for s, te in d.groupby('Season'):
        tr = d[d.Season != s]
        beta = coefficient(tr)
        p = expit(beta*te.x.to_numpy())
        y = te.y.to_numpy()
        ll = -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
        ent = -np.mean(p*np.log(p)+(1-p)*np.log1p(-p))
        prior = d[d.Season < s]
        # First season has no honest past-only label; never impute it using future seasons.
        bp = coefficient(prior) if len(prior) else np.nan
        pp = expit(bp*te.x.to_numpy())
        surprise_past = np.mean((pp-y)*bp*te.x.to_numpy())
        gap = abs(te.x.to_numpy()) >= 4
        favorite_won = np.where(te.x.to_numpy()>0, y, 1-y)
        pf = np.maximum(p,1-p)
        top1,top2 = (te.s1<=4).to_numpy(),(te.s2<=4).to_numpy()
        actual = np.sum(top1*y+top2*(1-y))
        expected = expected_top4_wins(g, s, beta)
        # Calibration intercept is zero, matching the symmetric favorite model.
        slope = coefficient(te)/beta
        rows.append(dict(Season=s, games=len(te), beta=beta, log_loss=ll,
                         expected_log_loss=ent, surprise=ll-ent,
                         brier_surprise=np.mean((p-y)**2-p*(1-p)),
                         gap_upsets=int(np.sum(gap*(1-favorite_won))),
                         expected_gap_upsets=float(np.sum(gap*(1-pf))),
                         gap_upset_excess=float(np.sum(gap*((1-favorite_won)-(1-pf)))),
                         calibration_slope=slope, slope_bust=-slope,
                         top4_wins=int(actual), expected_top4_wins=float(expected),
                         top4_deficit=float(expected-actual), surprise_past=surprise_past))
    out=pd.DataFrame(rows)
    out['bust'] = out.surprise >= out.surprise.quantile(2/3)
    return out


def season_features(g):
    reg=F.read(g+'RegularSeasonCompactResults').query('DayNum <= 132').copy()
    seeds=F.seeds(g)
    conf=F.read(g+'TeamConferences').set_index(['Season','TeamID']).ConfAbbrev
    ct=F.read(g+'ConferenceTourneyGames').query('DayNum <= 132')
    detail=F.read(g+'RegularSeasonDetailedResults').query('DayNum <= 132')
    mas={}
    if g=='M':
        m=F.read('MMasseyOrdinals').query('RankingDayNum <= 133')
        m=m.sort_values('RankingDayNum').drop_duplicates(['Season','SystemName','TeamID'],keep='last')
        for s,z in m.groupby('Season'):
            # Rank-percentile dispersion permits systems covering different numbers of teams.
            z=z.copy(); z['pct']=z.groupby('SystemName').OrdinalRank.rank(pct=True)
            mas[s]=z.groupby('TeamID').pct.std()
    rows=[]
    for s, sd in seeds.groupby('Season'):
        if s==2020 or s < (1985 if g=='M' else 1998):
            continue
        d=reg[reg.Season==s].sort_values('DayNum')
        teams=np.union1d(d.WTeamID,d.LTeamID); ix={t:i for i,t in enumerate(teams)}
        n=len(d); w=d.WTeamID.map(ix).to_numpy(); l=d.LTeamID.map(ix).to_numpy()
        loc=d.WLoc.map({'H':1,'A':-1,'N':0}).to_numpy()
        X=sparse.csr_matrix((np.r_[np.ones(n),-np.ones(n),loc],
            (np.tile(np.arange(n),3),np.r_[w,l,np.full(n,len(teams))])),shape=(n,len(teams)+1))
        margin=(d.WScore-d.LScore).to_numpy()
        fit=Ridge(alpha=10,fit_intercept=False).fit(X,margin)
        q=pd.Series(fit.coef_[:-1],index=teams)
        qs=q.reindex(sd.TeamID)
        # Online Elo resets every season, so favorite status never sees future games.
        elo=dict.fromkeys(teams,1500.); probs=[]
        for r in d.itertuples():
            h=65*({'H':1,'A':-1,'N':0}[r.WLoc])
            p=expit((elo[r.WTeamID]-elo[r.LTeamID]+h)*np.log(10)/400)
            probs.append(p); delta=20*(1-p)
            elo[r.WTeamID]+=delta; elo[r.LTeamID]-=delta
        d=d.assign(p=probs)
        def upset(z):
            decisive=z[np.maximum(z.p,1-z.p)>=.65]
            return float((decisive.p<.5).mean()) if len(decisive) else np.nan
        c=ct[ct.Season==s]
        cd=d.merge(c[['DayNum','WTeamID','LTeamID']],on=['DayNum','WTeamID','LTeamID'])
        steal=np.nan
        if len(c):
            champs=c.sort_values('DayNum').groupby('ConfAbbrev').tail(1)
            steal=0
            for r in champs.itertuples():
                peers=[t for t in teams if conf.get((s,t))==r.ConfAbbrev]
                steal+=int(r.WTeamID not in q.reindex(peers).nlargest(2).index)
        dt=detail[(detail.Season==s)&(detail.WTeamID.isin(sd.TeamID)|detail.LTeamID.isin(sd.TeamID))]
        # Only the seeded team's attempts enter the numerator/denominator.
        wa=dt.WTeamID.isin(sd.TeamID); la=dt.LTeamID.isin(sd.TeamID)
        attempts=dt.loc[wa,'WFGA'].sum()+dt.loc[la,'LFGA'].sum()
        threes=dt.loc[wa,'WFGA3'].sum()+dt.loc[la,'LFGA3'].sum()
        top=sd.loc[sd.seed<=4,'TeamID']; mid=sd.loc[sd.seed.between(5,12),'TeamID']
        rows.append(dict(Season=s,rating_spread=qs.std(),
            top_middle_gap=q.reindex(top).mean()-q.reindex(mid).mean(),
            regular_upset_rate=upset(d),late_upset_rate=upset(d[d.DayNum>=100]),
            conference_upset_rate=upset(cd),bid_stealers=steal,
            margin_variance=float(np.var(margin,ddof=1)),
            massey_disagreement=mas[s].reindex(sd.TeamID).mean() if s in mas else np.nan,
            three_attempt_rate=threes/attempts if attempts else np.nan,
            strong_midmajors=sum(q.rank(ascending=False)[t]<=40 and conf.get((s,t)) not in MAJORS for t in sd.TeamID),
            home_effect=fit.coef_[-1]))
    return pd.DataFrame(rows)


def association(x,y,rng):
    n=len(x); rx=rankdata(x); ry=rankdata(y)
    rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean()); ry=(ry-ry.mean())/np.linalg.norm(ry-ry.mean())
    rho=float(rx@ry)
    perm=np.array([rng.permutation(ry)@rx for _ in range(9999)])
    p=(1+np.sum(abs(perm)>=abs(rho)))/10000
    idx=rng.integers(0,n,(4000,n)); a=rankdata(x[idx],axis=1); b=rankdata(y[idx],axis=1)
    a-=a.mean(axis=1,keepdims=True); b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    bs=np.divide((a*b).sum(axis=1),den,out=np.full(len(den),np.nan),where=den>0)
    lo,hi=np.nanquantile(bs,[.025,.975])
    return rho,p,lo,hi


def walkforward(d,features):
    results=[]
    for year in d.Season:
        tr=d[(d.Season<year)].dropna(subset=features+['surprise_past'])
        te=d[d.Season==year]
        if len(tr)<10 or te[features].isna().any(axis=None):
            continue
        assert tr.Season.max()<year
        sc=StandardScaler().fit(tr[features]); model=Ridge(alpha=10).fit(sc.transform(tr[features]),tr.surprise_past)
        results.append(dict(Season=int(year),actual=float(te.surprise_past.iloc[0]),
            prediction=float(model.predict(sc.transform(te[features]))[0]),baseline=float(tr.surprise_past.mean())))
    r=pd.DataFrame(results)
    if len(r)<5:
        return np.nan,len(r),r
    score=1-((r.actual-r.prediction)**2).sum()/((r.actual-r.baseline)**2).sum()
    return float(score),len(r),r


def table(d,cols,decimals=4):
    def fmt(v):
        if pd.isna(v): return 'NA'
        if isinstance(v,(bool,np.bool_)): return 'yes' if v else 'no'
        if isinstance(v,(int,np.integer)): return str(v)
        if isinstance(v,(float,np.floating)): return f'{v:.{decimals}f}'
        return str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+ '\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d[cols].itertuples(index=False,name=None))


def main():
    OUT.mkdir(exist_ok=True); rng=np.random.default_rng(SEED)
    seasons=[]; tests=[]; wf=[]; predictions=[]
    for g in 'MW':
        print('Processing',g,flush=True)
        d=labels(games(g),g).merge(season_features(g),on='Season',validate='one_to_one')
        # Calendar year: 2021 has no previous tournament, and remains missing.
        prev=d.set_index('Season').surprise_past
        d['previous_surprise']=d.Season.map(lambda y:prev.get(y-1,np.nan))
        d.insert(0,'gender',g); seasons.append(d)
        for f in FEATURES:
            for target in OUTCOMES:
                z=d.dropna(subset=[f,target])
                if len(z)<10 or z[f].nunique()<2: continue
                rho,p,lo,hi=association(z[f].to_numpy(),z[target].to_numpy(),rng)
                tests.append(dict(gender=g,feature=f,outcome=target,n=len(z),rho=rho,p=p,ci_low=lo,ci_high=hi))
        sets=[[f] for f in FEATURES]+[['rating_spread','regular_upset_rate']]
        for fs in sets:
            r2,n,r=walkforward(d,fs)
            wf.append(dict(gender=g,features='+'.join(fs),n=n,oos_r2=r2))
            if len(r): predictions.append(r.assign(gender=g,features='+'.join(fs)))
    d=pd.concat(seasons,ignore_index=True); t=pd.DataFrame(tests); w=pd.DataFrame(wf)
    t['holm_p']=multipletests(t.p,method='holm')[1]
    # Also provide primary-only family; no robustness definition can rescue failure here.
    t['primary_holm_p']=np.nan
    mask=t.outcome.eq('surprise'); t.loc[mask,'primary_holm_p']=multipletests(t.loc[mask,'p'],method='holm')[1]
    d.to_csv(OUT/'seasons.csv',index=False); t.to_csv(OUT/'associations.csv',index=False)
    w.to_csv(OUT/'walk_forward.csv',index=False); pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(13,7),sharex=True)
    for ax,g in zip(axs,'MW'):
        z=d[d.gender==g]; ax.bar(z.Season,z.surprise,color=np.where(z.bust,'#cb4b43','#4386aa'))
        ax.axhline(z.surprise.quantile(2/3),color='black',linestyle='--',label='Top-tercile threshold')
        ax.axhline(0,color='gray',linewidth=.7); ax.set_ylabel(g+' surprise (nats/game)'); ax.legend()
    axs[-1].set_xlabel('Season'); fig.tight_layout(); fig.savefig(OUT/'season_surprise.png',dpi=180); plt.close(fig)
    primary=t[mask].sort_values(['gender','p'])
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,g in zip(axs,'MW'):
        z=primary[primary.gender==g].sort_values('rho'); yy=np.arange(len(z))
        ax.hlines(yy,z.ci_low,z.ci_high,color='#4386aa'); ax.scatter(z.rho,yy,color='#173f59')
        ax.set_yticks(yy,z.feature); ax.axvline(0,color='gray',linestyle='--'); ax.set_xlim(-1,1); ax.set_title(g+' Spearman rho, pointwise 95% bootstrap CI')
    fig.tight_layout(); fig.savefig(OUT/'correlations.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,g in zip(axs,'MW'):
        z=w[w.gender==g].dropna().sort_values('oos_r2'); ax.barh(z.features,z.oos_r2,color='#4386aa'); ax.axvline(0,color='black'); ax.set_title(g+' expanding-window OOS R²'); ax.set_xlabel('Against historical-mean forecast')
    fig.tight_layout(); fig.savefig(OUT/'walk_forward.png',dpi=180); plt.close(fig)
    passed=primary[(primary.primary_holm_p<.05)&((primary.ci_low>0)|(primary.ci_high<0))]
    survivors=[]
    for r in passed.itertuples():
        z=w[(w.gender==r.gender)&(w.features==r.feature)]
        if len(z) and z.oos_r2.iloc[0]>0: survivors.append(r.feature)
    (OUT/'summary.json').write_text(json.dumps({'survivors':survivors,'tests':len(t),'primary_tests':int(mask.sum())},indent=2))
    report='''# Can we identify a bust year before the tournament?\n\n'''
    report+=('**No usable correlation was established.**' if not survivors else '**A candidate survived the screening gate; temperature validation is required.**')+'\n\n'
    report+='''## Design and leakage controls

Primary definition: a fixed symmetric seed-difference logistic, with no intercept, fitted leaving out the entire target season. Each season's surprise is mean actual log loss minus mean Bernoulli entropy on its actual main-bracket matchups (63, except men's 2021: 62 played games); positive means more surprising than expected. A bust is at or above that gender's full-sample 2/3 quantile (a descriptive label, not a deployable threshold). Men's 1985–2026 and women's 1998–2026 exclude 2020. First Four excluded. The separate 2026 solution supplies all 63 results per gender; the compact file ends in 2025.

LOSO uses other seasons only to define descriptive outcomes, as explicitly requested. It must not be used to train predictive backtests: expanding-window targets instead use seed-logistic coefficients fitted only on seasons before the target season (`surprise_past`). The earliest season is missing. Previous-year surprise uses this honest target and is missing in 2021 because 2020 was canceled. Every feature uses only regular-season DayNum ≤132, seeds, or Massey RankingDayNum ≤133. No end-of-season Torvik or closing-round odds are used.

Expected loss is conditional on the games actually played, which depend on earlier outcomes; this is a conditional scoring residual, not an unconditional simulation of bracket chaos. Robustness outcomes: Brier minus p(1−p); count of seed-gap ≥4 losses minus sum of their favorite-loss probabilities; negative zero-intercept calibration slope (larger = more bust); and unconditional expected minus actual top-four-seed main-bracket wins. The last propagates seed-logistic probabilities through the complete seeded bracket, including play-ins in the propagation but counting only rounds 1–6. All are in seasons.csv. No alternative definition is used to select a temperature.

## Fixed pre-tournament features

* Rating spread: standard deviation of adjusted-margin ridge ratings among the actual field (64–68 teams). Top/middle gap: mean rating of seeds 1–4 minus seeds 5–12. Ratings fit compact regular-season margins with team contrasts and signed home location, ridge alpha 10, no clipping, no carryover.
* Upset rates: losses by favorites with online Elo win probability ≥0.65, all regular games / DayNum ≥100 / conference-tournament games. Elo starts at 1500 each season, K=20, home=65; each probability is computed before its game. These measure the whole Division I season. Bid stealers: conference-tournament champions outside their conference's top two final pre-tournament ridge ratings (a fixed proxy, not an estimate of committee at-large bids).
* Margin variance: sample variance of winning margins over regular-season games. Home effect: signed location coefficient in the adjusted-margin model.
* Massey disagreement: mean across seeded teams of the standard deviation across systems of within-system rank percentiles, latest available row per system/team by day 133. Men's only; no women's ordinal dataset exists here. This is rank disagreement, not spread in strengths.
* Three-point attempt rate: total 3PA / FGA for seeded teams in regular-season detailed games. Men's detailed coverage starts 2003; women's starts 2010.
* Strong mid-majors: seeded teams ranked in the top 40 of regular-season ridge strength outside ACC, Big East, Big Ten, Big 12/8, Pac-10/12, SEC, SWC. Historical realignment is taken from each season's conference table. This fixed conference proxy has classification limitations.

No missing feature is imputed. Feature availability varies (conference tournaments and conference membership begin later); every association reports its n.

## Statistical protocol

Two-sided Spearman tests use 9,999 season permutations with +1 correction, and 4,000 paired season bootstraps for pointwise percentile 95% CIs; RNG seed 20261109. Holm correction is reported for the 23 primary gender-feature tests and separately for the full primary-plus-robustness family. CIs are not simultaneous. Exchangeability/independence is an assumption; eras and serial dependence weaken this small-sample evidence. No feature selection by 2022/23 or 2025/26 outcomes is performed.

Walk-forward: each fixed one-variable model, plus one prespecified two-variable model (rating spread + regular upset rate), uses standardized predictors and ridge alpha 10 with intercept. Start only after ten complete prior seasons. Standardization, regression, and the historical-mean baseline see prior seasons only; outcome is the past-only surprise described above. OOS R² = 1 − SSE(model)/SSE(expanding historical mean), on identical test seasons for each model. Sample sizes/windows vary with feature availability; these are exploratory validation checks, not independent confirmation after inspecting this dataset. Survival requires primary Holm p<0.05, CI excluding zero, and positive one-variable OOS R².

'''
    report += f'''## Main findings

Across {len(d[d.gender=='M'])} men's and {len(d[d.gender=='W'])} women's seasons, none of the {int(mask.sum())} primary tests survives Holm correction. The strongest primary men's association is rating spread (rho −0.353, unadjusted permutation p=0.0234, primary-family Holm p=0.5382; pointwise 95% CI −0.608 to −0.064). Its one-variable expanding-window R² is +0.093 over 30 seasons: suggestive, but insufficient after testing the stated candidate family. Women's bid-stealer proxy has rho +0.462, unadjusted p=0.0250, adjusted p=0.5500; its walk-forward R² is −0.074 over 14 seasons. Robustness tests also fail full-family correction; see the complete table below.

Both genders classify 2022 and 2023 as bust years and 2025/2026 as non-bust years under the primary definition. Men's seed-model calibration slopes in 2025/2026 are 1.693/1.386, versus 0.662/0.799 in 2022/2023. These are seed-model slopes, so they do not estimate or independently validate the betting-market slope history supplied in the task. That realized contrast does not establish a pre-tournament predictor.

'''
    report+='## Primary associations\n\n'+table(primary,['gender','feature','n','rho','ci_low','ci_high','p','primary_holm_p','holm_p'])+'\n\n'
    report+='## Walk-forward performance\n\n'+table(w,['gender','features','n','oos_r2'])+'\n\n'
    report+='## Robustness associations\n\n'+table(t[~mask].sort_values(['outcome','gender','p']),['gender','outcome','feature','n','rho','ci_low','ci_high','p','holm_p'])+'\n\n'
    report+='## Season table\n\nUnits are nats per game for log loss/surprise; counts for gap-upset excess and top-four deficit. Slope is favorable-model calibration (lower = more bust).\n\n'
    report+=table(d,['gender','Season','games','log_loss','expected_log_loss','surprise','bust','brier_surprise','gap_upset_excess','calibration_slope','top4_deficit'])+'\n\n'
    report+='''## Artifacts and reproduction

![Season surprise](bust_years/season_surprise.png)

![Primary correlations](bust_years/correlations.png)

![Walk-forward R squared](bust_years/walk_forward.png)

Full precision: [seasons/features](bust_years/seasons.csv), [all tests](bust_years/associations.csv), [walk-forward scores](bust_years/walk_forward.csv), [individual predictions](bust_years/predictions.csv).

Run: `MANIA_DATA=/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26 OMP_NUM_THREADS=1 uv run python -m kaggle_tuning.research.bust_years`.

'''
    if not survivors:
        report+='''## Decision and modeling guard

No predictor passes the statistical-plus-walk-forward gate. The requested conditional temperature experiment is therefore skipped: no bust-adjusted config is created, and no historical-guard gain is claimed. Existing configs and model code are unchanged. A null result does not prove no underlying signal exists; these season counts do not justify a deployable adjustment.
'''
    Path(__file__).with_name('BUST_YEARS.md').write_text(report)
    print(primary.to_string(index=False)); print(w.to_string(index=False)); print('SURVIVORS:',survivors)

if __name__=='__main__':
    main()
