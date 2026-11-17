# Does reinforcement learning help the Kaggle Brier score? (issue #59)

**Short answer: no. Standard RL does not fit the 2025/2026 Kaggle task.** The parts of the RL toolkit
that do fit are online expert weighting methods with full feedback, such as Hedge. They are close
relatives of the walk-forward blend weight selection this branch already uses. This note runs no
experiment and claims no improvement. It ends with one test that could be run later.

## The task, stated precisely

- **Metric:** Brier score on the main-bracket games that are actually played, with men's and women's
  games pooled (63 + 63 in 2026). First Four games are excluded. Each pair's probability is fixed before
  the tournament begins. See [`KAGGLE_EDA.md`](KAGGLE_EDA.md) for the scoring population and the
  validated scorer in `kaggle_tuning/all_years.py`. Competition page:
  <https://www.kaggle.com/competitions/march-machine-learning-mania-2026>.
- **Years:** 2023, 2025 and 2026 used this metric. 2021/2022 used log loss with separate gender
  competitions. 2024 used a bracket Brier.
- **Leakage boundary:** features may use only regular-season games (DayNum ≤ 132), seeds, and men's
  Massey ordinals (≤ 133).

## Why RL does not fit

RL addresses sequential decisions. An agent's action changes the state of the environment, and the
reward may arrive only after several later steps. That is the Markov decision process setting in
Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press, 2018), ch. 3.
The Kaggle task has none of these features:

1. **One decision, made before the tournament.** All 132,133 probabilities are submitted before tip-off,
   and there is no action to take during the tournament.
2. **The submission does not change the outcomes.** The games are decided the same way whatever is
   submitted. There is no state transition and no exploration–exploitation trade-off, and no credit
   needs to be assigned across time steps.
3. **Feedback covers every prediction.** Once the games are played, the outcome of every scored pair is
   known. The bandit setting assumes the opposite: only the reward of the chosen action is seen
   (Auer, Cesa-Bianchi, Freund & Schapire, "The nonstochastic multiarmed bandit problem",
   *SIAM J. Computing* 32(1):48–77, 2002; Sutton & Barto ch. 2). With full feedback, the problem
   becomes ordinary supervised learning, or online learning with expert advice when it is repeated
   season after season (Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games*, Cambridge UP, 2006).
4. **The best submission is honest probability estimation.** The Brier score (Brier, "Verification of
   forecasts expressed in terms of probability", *Monthly Weather Review* 78(1):1–3, 1950) is strictly
   proper: in expectation, the best report is the forecaster's true probability (Gneiting & Raftery,
   "Strictly proper scoring rules, prediction, and estimation", *JASA* 102(477):359–378, 2007,
   doi:10.1198/016214506000001437). Searching for a "policy" therefore reduces to estimating
   well-calibrated win probabilities. That is the problem the existing logistic, ridge and
   gradient-boosting backtests already work on.

An RL agent trained in a simulated tournament would get its reward from the same game outcomes,
but with more variance and less sample efficiency than fitting the outcomes directly. With about
11 usable seasons × 126 scored games, data is the limiting factor.

## Where ideas from RL could fit

| Idea | Fit for Kaggle Brier? | Notes |
| --- | --- | --- |
| Hedge / exponential weights across model members (Freund & Schapire, "A decision-theoretic generalization of on-line learning…", *JCSS* 55(1):119–139, 1997) | **Plausible, untested** | This is learning with expert advice under full feedback, applied to Brier, which is bounded in [0,1]. `ext_blend.walk_forward_weight` already chooses a blend weight by the best past mean score (follow-the-leader on a grid), and Hedge is a smoothed form of that. With only about 11 seasons, regret bounds are loose, and the season-to-season shift (see `BUST_YEARS.md`) breaks their assumptions. |
| Contextual bandits (e.g. Li, Chu, Langford & Schapire, WWW 2010) | No | They assume partial feedback, but the Kaggle task gives full feedback. |
| MDP / policy-gradient / Q-learning | No | The task has no sequential state. |
| Updating probabilities during the tournament | Not for Kaggle | Kaggle locks submissions before the tournament. A live product could use this, but it is a different problem (outcome-driven Bayesian or Elo updates, not RL). |
| Rank- or prize-seeking gambles (e.g. the "Drake gamble" in the 2025 goto_conversion file, see `README.md`) | **Not a Brier improvement** | Leaderboard rank is not a proper objective. A deliberately overconfident pick can raise the chance of a top finish while making the *expected* Brier worse. Choosing such gambles is a question of game theory against other entrants, not of RL. Issue #59 asks to improve Brier, so this branch should not gamble. |
| Self-play / MCTS | No | Basketball outcomes are not a game that a forecaster plays. |

## A test that could be run (not run here)

If someone wants an RL-adjacent experiment, the honest version is this:

1. Fix the members before looking at results, for example the existing `final` men's members and
   `women_recommended` members.
2. Fix the Hedge learning rate η in advance, for example η = 8 per season-mean Brier. Do not tune it
   on 2025/2026.
3. Run the existing walk-forward: weights for season *y* use only seasons < *y*.
4. Compare it paired, on the same games, against the current blend. Apply the 0.0005 historical guard,
   and report men and women separately as well as the pooled 2025/2026 scores.

Until that has been run, Hedge has no measured benefit here.

## Limitations

- Concluding that RL does not fit is a structural argument about this metric. It does not show that
  every learning method with "reinforcement" in its name will fail.
- Bracket-pool formats (such as ESPN-style pick'em, or the 2024 bracket Brier) have different
  objectives, which this note does not analyse.
