# Kaggle dataset audit and EDA (issue #59)

Reproduce (reads `data/kaggle`, writes only [`kaggle_eda/summary.json`](kaggle_eda/summary.json)):

    MANIA_DATA=/abs/path/to/data/kaggle/mirror26 uv run --extra training python -m kaggle_tuning.research.kaggle_eda

Data: the `mirror26` snapshot (competition CSVs as of 2026-04-05), `sol26` (2026 solution),
and the 2025/2026 public leaderboards. All three are gitignored, so nothing here is committed.
Tests: `tests/test_kaggle_eda.py` (bracket-round mapping, past-only fitting).

**Scope.** These are descriptive checks and a seed-only reference model. Nothing here changes
a submission, and this report claims no Brier improvement.

## What is scored

| Item | Finding |
| --- | --- |
| Metric | 2025/2026: Brier score on games actually played, men + women pooled. 2021/2022 used log loss with separate competitions. 2024 used a bracket Brier (see `all_years.py`). The repo's scorer reproduces the leaderboard exactly (see `kaggle_tuning/README.md`). |
| Population | Stage 2 asks for 132,133 pairs; 126 are scored in 2026 (63 men + 63 women). That is 0.095%. |
| First Four | Excluded: men's compact results have 67 games per season, women's 67 since 2022 (63 before). Identified by both teams sharing a seed line (`W16a`/`W16b`), the same rule as `features.tourney_games`. |
| Rounds | Derived from seeds (`round_of`). Every season 2015–2026 has 32/16/8/4/2/1 games per round, except men's 2021: 62 games, because VCU–Oregon was a no-contest. |
| Orientation | Kaggle IDs put the lower TeamID first. Lower IDs won 51.7% (M) and 51.4% (W) of main-bracket games, which is close to random. ID order carries no useful signal. |
| Weight per round | Every game counts the same. Under the seed model, R64+R32 produce 72% (M) and 68% (W) of all squared error, and the Final Four plus title game produce about 5% (M) and 8% (W). |

## Leakage audit

| Check | Result | Implication |
| --- | --- | --- |
| Regular-season max DayNum | 132 (M and W) | Tournaments start at day 134 (M) and 135 (W), so the regular-season files are pre-tournament. |
| Conference tournaments | 100% of `*ConferenceTourneyGames` rows are also in the regular-season files (days 115–132). | They are already in the features; do not add them twice. |
| Massey ordinals | Men only, 2003–2026, max RankingDayNum 133 (402k rows on day 133). No women's file exists. | The `RankingDayNum <= 133` cutoff in `features.massey` is the latest pre-tournament value. No ordinals are available for women. |
| `MTeamCoaches` | LastDayNum goes up to 154. One row ends between day 133 and season end (a coach change during the postseason). | The file contains post-tournament information. Use only coach rows active on day 132. |
| Secondary tourneys (NIT etc.) | Start on day 133 (M) and 135 (W). | NIT participation is known at Selection Sunday, but NIT *results* happen during the NCAA tournament. Never use them as features. |
| Tournament `WLoc` | Men: 100% neutral. Women: 831/1,717 games (1998–2025) are home or away, because top seeds host the early rounds. The 2026 solution file has no venue column. | `WLoc` in the tournament results is recorded after the fact. A host feature would have to be rebuilt from the pre-announced host sites or seeds. 2026 venues are not in the data. |
| Detailed box scores | Men from 2003, women from 2010. | Box-score features cannot cover earlier seasons, so long backtests mix feature sets. |

## Seed-only walk-forward reference

Model: P(T1 wins) = sigmoid(b·(seed2 − seed1)), with no intercept and fitted symmetrically. It is fitted
separately per gender on **strictly earlier** seasons (all main-bracket games since 1985 for men, 1998 for women).
The metric is game-level Brier, applied uniformly. That was the official metric only in 2023, 2025 and 2026.
The 95% interval resamples that season's games (2,000 bootstraps).

| Season | n | Men | Women | Pooled | Pooled 95% |
| --- | --- | --- | --- | --- | --- |
| 2015 | 126 | 0.1704 | 0.1470 | 0.1587 | 0.127–0.192 |
| 2016 | 126 | 0.2028 | 0.1719 | 0.1873 | 0.151–0.227 |
| 2017 | 126 | 0.1765 | 0.1408 | 0.1586 | 0.127–0.194 |
| 2018 | 126 | 0.2063 | 0.1676 | 0.1870 | 0.149–0.226 |
| 2019 | 126 | 0.1735 | 0.1313 | 0.1524 | 0.124–0.185 |
| 2021 | 125 | 0.2291 | 0.1428 | 0.1856 | 0.149–0.227 |
| 2022 | 126 | 0.2190 | 0.1720 | 0.1955 | 0.158–0.237 |
| 2023 | 126 | 0.2047 | 0.1695 | 0.1871 | 0.151–0.224 |
| 2024 | 126 | 0.1915 | 0.1132 | 0.1524 | 0.123–0.185 |
| 2025 | 126 | 0.1528 | 0.1121 | 0.1324 | 0.109–0.158 |
| 2026 | 126 | 0.1653 | 0.1170 | 0.1412 | 0.113–0.171 |

A constant 0.5 scores 0.25. Across these 11 seasons, the seed-only mean is 0.1902 for men (SD 0.024),
0.1441 for women (SD 0.024) and 0.1671 pooled (SD 0.022).

**How to read the noise.** A single season's score has a 95% interval about ±0.03 wide. The pooled
seed-only score also moves by 0.022 (SD) from season to season. That is hundreds of times larger than the gains the
tuning branch reports (for example, 0.000077 historical for the women's recommendation). Absolute
season scores therefore cannot rank close configurations. Only *paired* comparisons on the same games
can, such as the existing 0.0005 guard. Even those cover only about 11 seasons × 126 games.
On the leaderboard, 69 (2025) and 82 (2026) teams fall within ±0.001 of the top-100 cutoff. Ranks near
that cutoff mostly reflect which games happened to be upsets.

**Men and women differ.** Women's games are much more predictable from seed alone in every season:
the women's mean is 0.046 lower. In the first round, men's 12–14 seeds won 36%, 21% and 14% of games
against 5, 4 and 3 seeds. Women's 13–16 seeds won 5.6%, 0%, 0% and 0.9% (1985/1998–2025). In the pooled
metric, most of the room to improve is in the men's games.

### By round (2015–2026, seed-only)

| Round | Men n | Men Brier | Men fav. win | Women n | Women Brier | Women fav. win |
| --- | --- | --- | --- | --- | --- | --- |
| R64 | 351 | 0.174 | 72.1% | 352 | 0.110 | 83.5% |
| R32 | 176 | 0.189 | 73.9% | 176 | 0.163 | 76.1% |
| S16 | 88 | 0.230 | 68.2% | 88 | 0.184 | 75.0% |
| E8 | 44 | 0.244 | 63.6% | 44 | 0.185 | 77.3% |
| F4 | 22 | 0.181 | 87.5%* | 22 | 0.248 | 53.8%* |
| NC | 11 | 0.216 | 83.3%* | 11 | 0.227 | 66.7%* |

\* The favourite rate leaves out games between equal seeds, so these cells have fewer than 10 games each.

Seed alone tells the teams apart less well in later rounds. Most of the error that a better team-strength
model could remove is in the S16 and E8 games, plus the men's R64.

## Women's home-court hypothesis (descriptive, not a tested feature)

Of 767 non-neutral women's tournament games (24 seasons from 1999 to 2025, 6–63 per season; none in 2015 or 2021), the home team won 82.3%.
The seed-only model, fitted on earlier seasons, expected 79.0%. The excess of +3.3 points has a
season-block bootstrap 95% interval of +0.7 to +6.4 points. However, the home team was the better seed
in 94% of these games. The excess may therefore reflect a seed-model misfit for top-four seeds rather
than a venue effect. The current models do not use tournament venue. A real test would need pre-deadline
host-site data (not in the Kaggle files, and missing for 2026) and a walk-forward paired Brier
comparison against the same model without the host term. **This report has not run that test.**

## Not covered here

- Neither the betting-market files nor the external ratings (`ext_*`) are audited here; see `EXT_SOURCES.md` and `TORVIK_PROVENANCE.md`.
- No feature is proposed, and no model or submission was changed.
- 2020 has no tournament. The 2021 bubble tournament (all games at neutral sites, including women's) is included without special handling.
