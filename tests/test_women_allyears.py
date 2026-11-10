import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kaggle_tuning.research import women_allyears as W


class ConsensusTests(unittest.TestCase):
    def tearDown(self):
        W.ratings.cache_clear()

    def test_ratings_ignore_games_after_cutoff(self):
        rows = pd.DataFrame({'Season': [2014]*4, 'DayNum': [100,110,132,133],
            'T1': [1,1,2,1], 'T2': [2,3,3,2], 'loc': [0,1,-1,0],
            'margin': [10,15,5,10000]})
        for side in ['T1_', 'T2_']:
            for col,val in [('FGA',60),('OR',10),('TO',10),('FTA',20)]:
                rows[side+col] = val
        with patch.object(W.F, 'team_games', return_value=rows):
            a = W.ratings()
        W.ratings.cache_clear()
        with patch.object(W.F, 'team_games', return_value=rows.iloc[:3]):
            b = W.ratings()
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.isfinite(a[['bt','adj_eff']]).all().all())
        self.assertGreater(a.bt.iloc[0], a.bt.iloc[2])

    def test_classifier_only_uses_prior_seasons(self):
        table = pd.DataFrame({'Season':[2014,2014,2015,2015], 'TeamID':[1,2,1,2], 'seed':[1,2,1,2]})
        games = pd.DataFrame({'Season':[2014,2014,2015], 'T1':[1,2,1], 'T2':[2,1,2], 'margin':[1,-1,1]})
        with patch.object(W, 'ratings', return_value=table[['Season','TeamID']]), \
             patch.object(W.F, 'tourney_games', return_value=games), \
             patch('kaggle_tuning.backtest.test_pairs', return_value=games.iloc[2:]), \
             patch.object(W, 'LogisticRegression') as lr:
            lr.return_value.fit.return_value.predict_proba.return_value = np.array([[.2,.8]])
            W.predict({'consensus_features':['seed']}, 2015, table)
        self.assertEqual(len(lr.return_value.fit.call_args.args[1]), 2)


class TemperatureTests(unittest.TestCase):
    def test_post_calibrates_only_outer_model_only_pairwise_predictions(self):
        from kaggle_tuning import backtest as B
        from kaggle_tuning.configs.women_allyears import RECOMMENDED
        from kaggle_tuning.research import women_temperature as T
        cfg = {**RECOMMENDED, 'market_w': {}}
        with patch.object(T,'temperature',return_value=.9) as temperature:
            p=B._post(cfg,'W',2021,np.array([.2,.8]))
            temperature.assert_called_once()
            np.testing.assert_allclose(p,1-p[::-1])
            B._post(cfg,'W',2024,np.array([.2,.8]))
            B._post(cfg,'M',2021,np.array([.2,.8]))
            self.assertEqual(temperature.call_count,1)
        with patch.object(T,'temperature',side_effect=AssertionError('market must retain calibration')), \
             patch.object(B,'test_pairs',return_value=pd.DataFrame({'T1':[1], 'T2':[2]})), \
             patch.object(B.MK,'market',return_value=pd.DataFrame({'T1':[1],'T2':[2],'mkt':[.8]})):
            B._post(RECOMMENDED,'W',2025,np.array([.8]))

    def test_temperature_training_excludes_target_and_future(self):
        from kaggle_tuning.research import women_temperature as T
        calls=[]
        def prior(cfg,year):
            calls.append(year)
            self.assertNotIn('women_model_temperature',cfg)
            return pd.DataFrame({'p':[.2,.8], 'y':[0,1]})
        cfg={'models':[], 'sharpen':{'W':1.1}, 'women_model_temperature':{'start':2014,'penalty':.01}}
        with patch.object(T,'Memory') as memory:
            memory.return_value.cache.return_value=prior
            a=T.temperature(cfg,2018)
        self.assertEqual(calls,[2014,2015,2016,2017])
        self.assertTrue(.7<=a<=1.5)

if __name__ == '__main__':
    unittest.main()
