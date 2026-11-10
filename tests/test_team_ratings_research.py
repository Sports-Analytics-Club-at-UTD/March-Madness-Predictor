"""Synthetic check: adding late games or changing future tournaments cannot change past features."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from kaggle_tuning.research import team_ratings as R

class RatingTests(unittest.TestCase):
    def test_cutoff_and_walk_forward(self):
        rows=[]
        for year in (2013,2014):
            for day in range(10,130,10):
                for a,b in ((1,2),(2,3),(3,1)):
                    for t,o,score,opp,loc in ((a,b,80,65,1),(b,a,65,80,-1)):
                        rows.append(dict(Season=year,DayNum=day,T1=t,T2=o,margin=score-opp,loc=loc,
                          T1_Score=score,T2_Score=opp,T1_FGA=60,T2_FGA=60,T1_OR=10,T2_OR=10,
                          T1_TO=12,T2_TO=12,T1_FTA=20,T2_FTA=20))
        games=pd.DataFrame(rows)
        tour=pd.DataFrame(dict(Season=[2013,2014],T1=[1,1],T2=[2,2],margin=[10,10]))
        def fit(d,t):
            R.rating_table.cache_clear()
            with patch.object(R.F,'team_games',return_value=d),patch.object(R.F,'tourney_games',return_value=t):
                return R.rating_table('M').copy()
        expected=fit(games,tour)
        late=games.iloc[:2].assign(DayNum=140,T1_Score=200,margin=135)
        changed=tour.assign(margin=[10,-100])
        actual=fit(pd.concat([games,late]),changed)
        pd.testing.assert_frame_equal(expected,actual)
        self.assertTrue(np.isfinite(actual.to_numpy()).all())
        self.assertLess(abs(actual.rt_off1.mean()),1e-8)
        R.rating_table.cache_clear()
if __name__=='__main__':unittest.main()
