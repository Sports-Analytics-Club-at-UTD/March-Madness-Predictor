# kaggle_tuning

Leak-free walk-forward backtests for Kaggle March Machine Learning Mania (2025/2026 metric:
Brier on the 63 men's + 63 women's tournament games, First Four excluded).

**Submitting:** see [Kaggle submission](#kaggle-submission-submitpy) below. The rules do not
name betting odds, which the recommended blend uses. Read that flag before uploading.

Data (gitignored) lives in `data/kaggle/` (override with `MANIA_DATA=/abs/path/to/mirror26`):
- `mirror26/` – competition CSVs as of 2026-04-05 (Kaggle dataset muhammadtojiddinov/march-machine-learning-mania-2026)
- `sol26/` – 2026 solution file (brisamarina/2026-march-machine-learning-mania-solution-file)
- `market25/submission.csv`, `market26/submission.csv` – pre-deadline betting-odds probabilities (goto_conversion)
- `lb/{2024,2025,2026}/` – final leaderboards

Scorer is validated: the goto_conversion file scores exactly 0.12584 (2026) and 0.10921 (2025, with its Drake gamble), matching the leaderboard.

- `python -m kaggle_tuning.grid kaggle_tuning/configs/r6.py` – run configs in parallel. Columns: hist (2015–2024 model-only mean, no market), v21-24 (overfitting check), 2025, 2026 (+ men/women).
- `python -m kaggle_tuning.report --og-dir ...` – step-by-step comparison vs the original pipeline.

Leaderboard cutoffs: top-100 = 0.11078 / 0.12430, top-20 = 0.10921 / 0.12016 (2025 / 2026).

Women's research (`configs/women_recommended.py`):

- Leaves men on `final`; women combine 50% logistic (ridge alpha 0.1), 10% XGBoost,
  and 40% logistic using the original pipeline's trimmed efficiency, four factors,
  quality wins and schedule strength. Ensembles combine log odds. Women's final
  probabilities receive 1.1 log-odds sharpening; the market weight remains 0.5.
- Overall 2025/2026: **0.10838 / 0.12142**; women: **0.09061 / 0.09826**.
  Historical mean improves 0.000077, hW improves 0.000147, and v21–24 improves
  0.000031. The aggressive variant increases the original member to 50% and
  scores 0.10814 / 0.12110, but historical mean rises 0.000292 and v21–24 rises
  0.000447, close to the shared 0.0005 guard.
- Women's model alone scores 0.09415 / 0.09661. Market weights 0.4, 0.5, 0.6
  give overall two-year means 0.11484, 0.11490, 0.11503. Keep 0.5 by the existing
  equal-weight rule: there is no historical market evidence to select 0.4.
  `women_market_sensitivity.py` reproduces these checks; results are stored in
  `women_market_results.json`.
- `women_results.json` records all 58 configurations and per-season checks.
  `configs/women.py`, `women_round2.py`, `women_round3.py`, and
  `women_sensitivity.py` reproduce the research grids. Full original-model
  replacement, isotonic calibration and stronger sharpening failed the guard.
  No external ratings were used; joint-gender training was not tested.

Run the recommendation with the ordinary grid runner. `python -m
kaggle_tuning.women_grid <config.py>` additionally caches fits under `/private/tmp`,
keyed by implementation, full configuration, and data-file metadata, and saves a
`.results.json` beside the config. Set `MANIA_DATA` and `GRID_JOBS=5` as above.
The women's tests check calibration symmetry, target-season exclusion, and the
DayNum 132 cutoff. The original `configs/final.py` still reproduces 0.1099 / 0.1227.

## Kaggle submission (`submit.py`)

> **Market odds are not named in the rules.** The recommended configuration blends
> goto_conversion betting-market probabilities into the predictions (men 0.85, women 0.5, both
> after 1.2 log-odds sharpening of the market). The 2026 rules do not mention betting odds. They
> allow External Data that is "publicly available and equally accessible to use by all
> Participants of the Competition for purposes of the competition at no cost to the other
> Participants" or of "minimal cost" (§2.6). They also allow information from "human prediction
> of the validation dataset or test data records" (§3.4.b). goto_conversion is free on PyPI, and
> it publishes its submission, so it appears to qualify. That is our reading, not an explicit
> ruling. If the team or the host decides otherwise, submit `--model-only`: 2026 Brier 0.12253
> instead of 0.12142.

### Rules that matter (2026 edition)

Source: the Overview, Data and Rules pages of
https://www.kaggle.com/competitions/march-machine-learning-mania-2026. Kaggle renders those pages
client-side. The quotes come from a verbatim copy saved on 2026-03-09 in the public repository
`brownrv/mmlm2026` (`docs/march-machine-learning-mania-2026 - Overview and Data and Rules.docx`).
The 2027 page does not exist yet (404 on 2026-11-17). Re-read it when it opens.

- **One file for both tournaments.** Columns are `ID,Pred`. The ID format is `SSSS_XXXX_YYYY`:
  season, then the lower TeamID, then the higher TeamID. Pred is P(lower-ID team wins). Men's
  IDs are 1000–1999 and women's are 3000–3999.
- **Every possible matchup** between D-I teams of that season gets a row, not only tournament
  teams. `SampleSubmissionStage1.csv` covers 2022–2025 (519,144 rows) and only feeds the
  pre-tournament leaderboard. `SampleSubmissionStage2.csv` is the file to submit: 132,133 rows
  for 2026 (66,430 men's and 65,703 women's). Rows that are never played are never scored.
  `submit.py` writes 0.5 for any pair where a team is unseeded.
- **Metric:** Brier score, which is the MSE of Pred against the 0/1 outcome, over games actually
  played. The First Four is not scored: our scorer uses 63 + 63 games and reproduces the
  leaderboard exactly. Before tip-off, a correctly formatted 2026 file shows 0.0.
- **Deadline pattern:** the competition opens in mid-February. Seeds come out on Selection
  Sunday (DayNum 132; 2026-03-15). The final submission deadline is the Thursday that the
  Round of 64 starts, at 4 PM UTC (2026-03-19 16:00 UTC). Kaggle refreshes the data "at least
  once in advance of the deadline" (the 2026 data was complete through Feb 4 when it opened).
- **Limits:** 5 submissions per day. Select **one** Final Submission yourself: "Do not rely on
  automatic selection." Teams can have at most 5 people. No private code or data sharing
  outside the team. Winners must post a reproducible write-up with a code repository.

### Entry point

```bash
export MANIA_DATA=/abs/path/data/kaggle/mirror26   # competition CSVs incl. SampleSubmissionStage2.csv
uv sync --extra training
uv run --extra training python -m kaggle_tuning.submit --season 2026 --out submission.csv \
    [--market path/to/goto_submission.csv | --model-only] [--score solution.csv]
```

The script fits `configs/women_recommended.RECOMMENDED` (men = `configs/final.BASE`). Every
competition CSV is first cut to what was known on Selection Sunday of `--season`: regular season
DayNum ≤ 132, Massey RankingDayNum ≤ 133, and NCAA results only from earlier seasons. Market odds
are not filtered. They must come from before the submission deadline, not from before Selection
Sunday: the 2026 file is goto-conversion 4.0.1 from 2026-03-17, two days before the deadline. The script then predicts
all seeded × seeded pairs, writes the file, and re-reads it to check that the IDs and order
exactly match the sample, that there are no duplicates, and that every Pred is finite and in
[0, 1]. It writes `<out>.provenance.json` with the config, git SHA and dirty flag, package
versions, the sha256 of every input CSV, the sample and market files, and the submission
itself. It refuses to run without at least 64 seeds per gender, or when the market file lacks
odds for a seeded pair. It takes about a minute.

Reproduction (2026-11-17): the 2026 run scores **0.12142** (M 0.14458, W 0.09826) against
`sol26`, which matches the `women_recommended` row above and would have placed #34 of 3,462.
`--model-only` scores 0.12253 (#56). The 2025 run (Stage 1 rows for 2025 as `--sample`, with
the solution built from `MNCAATourneyCompactResults` without the First Four) scores 0.10838,
which also matches. The scorer reproduces goto_conversion's 0.12584 on `market26/submission.csv`.

### Final submission checklist (next season)

1. When the competition opens, read its Rules, Data and Evaluation pages. Check the ID format,
   the metric, the deadline, and anything on external data or betting odds. If odds are
   excluded, use `--model-only`.
2. After Selection Sunday, download the refreshed competition data into a new `mirror<YY>/`.
   Check that `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` contain the new season (68 each),
   and that `MMasseyOrdinals.csv` has RankingDayNum 133 rows for it.
3. Get the pre-deadline market odds: the new `goto-conversion` release's submission, in Kaggle
   format. Pass it with `--market`, or add it to `market.PATHS`. Note its source and date.
4. Run `submit.py --season <YYYY> --out submission.csv --market ...`. Confirm "validation
   passed", the row count equals the Stage 2 sample, and roughly 2,278 modelled pairs per gender.
5. Run the tests: `uv run python -m unittest discover -s tests`.
6. Archive `submission.csv` and `submission.provenance.json` somewhere outside git, for example
   on the team drive.
7. A human uploads the file before Thursday 4 PM UTC and **selects it as the Final Submission**.
   Keep a `--model-only` file ready as a fallback.
