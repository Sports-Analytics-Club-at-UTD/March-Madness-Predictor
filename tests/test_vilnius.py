"""Small source-fidelity and leakage checks, without fitting full tournament models."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kaggle_tuning import vilnius as V
from kaggle_tuning import vilnius_experiment as E
from kaggle_tuning import features as F


class VilniusTests(unittest.TestCase):
    def test_overtime_and_swap(self):
        row = {'Season': 2024, 'DayNum': 100, 'WTeamID': 1101, 'LTeamID': 1102, 'NumOT': 2}
        row.update({s+c: (100 if s == 'W' else 80) for s in ('W','L') for c in ['Score']+F.BOX})
        d = V.prepare(pd.DataFrame([row]))
        np.testing.assert_allclose(d.margin, [16,-16])
        np.testing.assert_allclose(d.T1_FGA, [80,64])
        self.assertEqual(d.T1.tolist(), [1101,1102])
        self.assertEqual(len(V.feature_names()),25)
        self.assertEqual(len(V.feature_names(True)),29)

    def test_gaussian_glm_equivalence(self):
        import statsmodels.api as sm
        pairs = pd.DataFrame({'T1':[1,1,2,3,2,4], 'T2':[2,3,3,4,4,5], 'margin':[2.,4.,3.,6.,2.,1.]})
        rev = pairs.rename(columns={'T1':'T2','T2':'T1'}).assign(margin=-pairs.margin)
        d = pd.concat([pairs,rev],ignore_index=True)
        sd = {1,2}
        kept = sd | set(d.loc[(d.margin>0)&d.T2.isin(sd),'T1'])
        dt = d[d.T1.isin(kept)|d.T2.isin(kept)].copy()
        for c in ('T1','T2'):
            dt[c+'_TeamID'] = dt[c].where(dt[c].isin(kept),0).map(lambda x:f'{x:04}')
        fit = sm.GLM.from_formula('margin~-1+T1_TeamID+T2_TeamID',dt,family=sm.families.Gaussian()).fit()
        expected = fit.params[fit.params.index.str.startswith('T1_')].to_numpy()
        np.testing.assert_allclose(V.quality(d,sd).quality,expected,atol=1e-7)

    def test_weight_fit_excludes_targets(self):
        seen=[]
        def fake_get(name,year):
            seen.append(year)
            return {'M':np.array([.7,.3]),'W':np.array([.7,.3])}
        with patch.object(E,'get',side_effect=fake_get), patch.object(E,'test_pairs',return_value=pd.DataFrame({'y':[1,0]})):
            w25=E.weights(('L','X','V'),2025,'M')
            w26=E.weights(('L','X','V'),2026,'M')
            self.assertLessEqual(max(seen),2024)
            np.testing.assert_allclose(w25,w26)
            seen.clear()
            E.weights(('L','X','V'),2018,'M')
            self.assertLess(max(seen),2018)

    def test_loso_uses_only_past_seasons(self):
        import xgboost as xgb
        from kaggle_tuning import backtest as B
        from types import SimpleNamespace
        rows = pd.DataFrame({'Season': [2003]*4+[2004]*4+[2005]*4,
                             'T1': [1101]*12, 'T2': [1102]*12,
                             'margin': [1,2,3,4,5,6,7,8,999,999,999,999]})
        calls = []
        def fake_matchup(pairs):
            out = pairs.copy()
            for col in V.feature_names():
                out[col] = 1.
            return out
        def fake_train(params, data, **kwargs):
            calls.append(data.get_label())
            return SimpleNamespace(predict=lambda data: np.zeros(data.num_row()))
        test = pd.DataFrame({'Season':[2005], 'T1':[1101], 'T2':[1102]})
        V.predict_year.cache_clear()
        with patch.object(F,'read',return_value=rows), patch.object(V,'prepare',side_effect=lambda x:x), \
             patch.object(V,'matchup',side_effect=fake_matchup), patch.object(B,'test_pairs',return_value=test), \
             patch.object(xgb,'train',side_effect=fake_train), \
             patch.object(V,'UnivariateSpline',return_value=lambda x: np.full(len(x),.5)):
            result = V.predict_year(2005,'{}')
        self.assertEqual(len(calls),2)
        self.assertTrue(all(np.max(y)<999 for y in calls))
        np.testing.assert_allclose(result['M'],[.5])
        V.predict_year.cache_clear()

    def test_score_weights_games_not_genders(self):
        with patch.object(E, 'HIST', [2021]), patch.object(E, 'VERIFY', [2021]), \
             patch.object(E, 'test_pairs', side_effect=lambda g,y: pd.DataFrame({'y': [0,0] if g=='M' else [0]})):
            result = E.summary({2021: {'M':np.ones(2), 'W':np.zeros(1)}})
        self.assertAlmostEqual(result['hist'], 2/3)

    def test_gender_flag(self):
        with patch.object(V,'team_table',return_value=pd.DataFrame({'Season':[2024]*4,'TeamID':[1101,1102,3101,3102], 'seed':[1,2,1,2], 'elo':[1000]*4})):
            d=V.matchup(pd.DataFrame({'Season':[2024,2024],'T1':[1101,3101],'T2':[1102,3102]}))
        self.assertEqual(d.men_women.tolist(),[1,0])


if __name__=='__main__':
    unittest.main()
