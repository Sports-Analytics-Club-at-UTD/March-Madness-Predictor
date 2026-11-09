# Torvik source audit

Downloaded `jswanson/barttorvik-t-rank-2015-2026` via Kaggle CLI on 2026-11-09,
without executing dataset contents. Stored separately in
`/Users/spro/code/March-Madness-Predictor/data/kaggle/ext_mens/`.

**Excluded from every experiment.** Historical records demonstrably include
postseason games: `mens_2015.csv` starts Kentucky 38–1, Duke 35–4,
Wisconsin 36–4; `mens_2025.csv` starts Houston 35–5, Duke 35–4,
Auburn 32–6. These are not Selection Sunday records. The 2026 file starts
Michigan 14–0, Vanderbilt 15–0, Iowa State 15–0, suggesting a different,
in-season snapshot; that does not repair the contaminated training years.
There is no verified pre-deadline publication/snapshot chain for the full series.

SHA256:

- mens_2015.csv: `32ca7b229cb1dcb5767ce37847b45b93c4aff2af4e4f2567816928c708405c32`
- mens_2025.csv: `919ec8fe18734d32cf834cbdcf9be2b28430e7d5cb4f95f9325ea3c6e84a465a`
- mens_2026.csv: `955059df0d127c06d48f51318cb210a0eda5913cb571fc185a2a2696e722be54`

Primary author guidance points toward a separate future source:
[Bart Torvik's own comment](https://adamcwisports.blogspot.com/p/data.html)
recommends the T-Rank Time Machine for actual ratings the day after Selection
Sunday. No such time-machine export was used here.
