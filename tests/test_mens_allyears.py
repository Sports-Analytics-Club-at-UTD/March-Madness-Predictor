"""Small synthetic leakage and symmetry checks for the men's research models."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from kaggle_tuning import mens_allyears as M


class MensAllYearsTests(unittest.TestCase):
    def test_regular_features_ignore_later_games_and_ordinals(self):
        games = pd.DataFrame({
            'Season':[2003]*4,'DayNum':[50,50,90,90],
            'T1':[1,2,2,3],'T2':[2,1,3,2], 'loc':[1,-1,0,0],
            'margin':[10,-10,5,-5],'T1_Score':[80,70,75,70],
            'T1_FGA':[60]*4,'T1_OR':[10]*4,'T1_TO':[12]*4,'T1_FTA':[20]*4,
            'T2_FGA':[60]*4,'T2_OR':[10]*4,'T2_TO':[12]*4,'T2_FTA':[20]*4})
        ranks = pd.DataFrame({'Season':[2003]*3,'TeamID':[1,2,3],
                              'SystemName':['POM']*3,'RankingDayNum':[90]*3,'OrdinalRank':[1,2,3]})
        try:
            with patch.object(M.F,'team_games',return_value=games),patch.object(M.F,'read',return_value=ranks):
                M.table.cache_clear()
                before=M.table(100)
            later=games.iloc[:2].copy();later['DayNum']=101;later['margin']=999
            later_r=ranks.copy();later_r['RankingDayNum']=101;later_r['OrdinalRank']=999
            with patch.object(M.F,'team_games',return_value=pd.concat([games,later])),patch.object(M.F,'read',return_value=pd.concat([ranks,later_r])):
                M.table.cache_clear()
                pd.testing.assert_frame_equal(before,M.table(100))
        finally:
            M.table.cache_clear()

    def test_model_temperature_precedes_market_blend(self):
        from kaggle_tuning import backtest as B
        from scipy.special import expit, logit
        p=np.array([.8,.2])
        mk=pd.DataFrame({'T1':[1,2],'T2':[2,1],'mkt':[.7,.3]})
        cfg={'model_sharpen':{'M':.9},'market_w':{'M':.85},'mkt_sharpen':1.2}
        with patch.object(B,'test_pairs',return_value=mk[['T1','T2']]),patch.object(B.MK,'market',return_value=mk):
            actual=B._post(cfg,'M',2025,p)
        expected=expit(.15*.9*logit(p)+.85*1.2*logit(mk.mkt.values))
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_margin_training_excludes_target_and_preserves_symmetry(self):
        from kaggle_tuning import backtest as B
        rows=[];teams=[]
        for year in range(2003,2007):
            teams.extend([{'Season':year,'TeamID':t,'q':float(t)} for t in [1,2,3]])
            for a,b,margin in [(1,2,-4),(2,3,3),(1,3,-7)]:
                rows.extend([{'Season':year,'T1':a,'T2':b,'margin':margin},
                             {'Season':year,'T1':b,'T2':a,'margin':-margin}])
        games=pd.DataFrame(rows)
        te=pd.DataFrame({'Season':[2006,2006],'T1':[1,3],'T2':[3,1]})
        opts={'features':['q'],'alpha':1}
        with patch.object(M,'margin_table',return_value=pd.DataFrame(teams)),patch.object(M.F,'tourney_games',return_value=games),patch.object(B,'test_pairs',return_value=te):
            p=M.predict_margin(opts,2006)
            games.loc[games.Season>=2006,'margin']=100000
            np.testing.assert_allclose(p,M.predict_margin(opts,2006))
            np.testing.assert_allclose(p[0]+p[1],1,atol=1e-12)


if __name__=='__main__':unittest.main()
