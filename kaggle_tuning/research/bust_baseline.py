"""Reproduce unchanged reference scores; bounded to three worker processes."""
import json
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from kaggle_tuning import all_years as A
from kaggle_tuning.configs.allyears import EXT
from kaggle_tuning.configs.combined import BEST

OUT=Path(__file__).with_name('bust_years')
YEARS=[2015,2016,2017,2018,2019,2021,2022,2023,2024,2025,2026]


def main():
    result={}
    for name,cfg in [('allyears_EXT',EXT),('combined_BEST',BEST)]:
        print('Reference',name,flush=True)
        pairs=dict(zip(YEARS,Parallel(n_jobs=3)(delayed(A.all_pair_predictions)(cfg,y) for y in YEARS)))
        scores=[A.score_year(cfg,y,pairs[y],{s:p for s,p in pairs.items() if s<y}) for y in YEARS if y>=2021]
        briers={y:float(np.mean(np.concatenate([(p.dropna(subset=['y']).p-p.dropna(subset=['y']).y).to_numpy()**2 for p in pairs[y].values()]))) for y in YEARS if y<2025}
        result[name]={'config':cfg,'all_years':scores,'historical_model_brier':briers,
                      'hist':float(np.mean(list(briers.values()))),
                      '2015_2019':float(np.mean([briers[y] for y in range(2015,2020)]))}
        (OUT/'baseline.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result[name],indent=2),flush=True)
    lines = ['\n## Unchanged reference-model reproduction\n',
             'Scores below were recomputed through `all_years.all_pair_predictions` and `all_years.score_year`, with three workers. Hist and 2015–2019 are model-only mean game Brier across seasons (no external blend). No candidate config was tested because the research gate failed.\n',
             '| Config | hist 2015–2024 | 2015–2019 |', '| --- | --- | --- |']
    for name,r in result.items():
        lines.append(f"| {name} | {r['hist']:.8f} | {r['2015_2019']:.8f} |")
    lines += ['\n| Config | Season | Metric | Score / rank |', '| --- | --- | --- | --- |']
    for name,r in result.items():
        for row in r['all_years']:
            score = (f"{row['all'][0]:.5f} / #{row['all'][1]}" if 'all' in row else
                     f"M {row['M'][0]:.5f} / #{row['M'][1]}; W {row['W'][0]:.5f} / #{row['W'][1]}")
            lines.append(f"| {name} | {row['year']} | {row['metric']} | {score} |")
    lines += ['\nExact unchanged config dictionaries and full-precision scores: [baseline.json](bust_years/baseline.json).',
              'No proposed change failed the 2015–2019 Brier guard: the statistical screening failed first, and no temperature was fitted.']
    report=OUT.parent/'BUST_YEARS.md'
    text=report.read_text().split('\n## Unchanged reference-model reproduction')[0]
    report.write_text(text+'\n'.join(lines)+'\n')

if __name__=='__main__': main()
