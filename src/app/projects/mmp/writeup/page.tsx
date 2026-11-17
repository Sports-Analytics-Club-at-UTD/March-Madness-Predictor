import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { ThemeToggleButton } from "@/components/theme-toggle-button";
import { CodeBlock } from "@/components/writeup/code-block";
import { BayesianFigure, LeaderboardFigure, MarketSensitivityFigure, SeedNoiseFigure } from "@/components/writeup/figures";
import { marketLift } from "@/lib/writeup-data";

export const metadata: Metadata = {
  title: "How the March Madness Predictor works",
  description: "Data sources, model choices, evaluation and dead ends behind the Sports Analytics Club at UTD March Madness Predictor.",
};

const SECTIONS = [
  { id: "overview", label: "Two problems" },
  { id: "data", label: "Data and cutoffs" },
  { id: "bracket-model", label: "Bracket model" },
  { id: "kaggle-model", label: "Kaggle ensemble" },
  { id: "evaluation", label: "Evaluation" },
  { id: "dead-ends", label: "What didn't work" },
  { id: "reproduce", label: "Reproduce it" },
];

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-24 border-t border-border pt-12 first:border-0 first:pt-0">
      <p className="font-mono text-xs uppercase tracking-widest text-accent">{eyebrow}</p>
      <h2 id={`${id}-title`} className="mt-2 text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">{title}</h2>
      <div className="mt-5 space-y-4 text-[0.95rem] leading-7 text-foreground/85">{children}</div>
    </section>
  );
}

function H3({ children }: { children: ReactNode }) {
  return <h3 className="pt-4 text-lg font-semibold tracking-tight text-foreground">{children}</h3>;
}

function C({ children }: { children: ReactNode }) {
  return <code className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.82em] text-foreground">{children}</code>;
}

function Stat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-2xl font-semibold tabular-nums text-card-foreground">{value}</div>
      <div className="mt-1 text-xs leading-snug text-muted-foreground">{detail}</div>
    </div>
  );
}

function Callout({ title, children }: { title: string; children: ReactNode }) {
  return (
    <aside className="rounded-xl border border-primary/30 bg-primary/5 p-4 text-sm leading-6">
      <div className="font-medium text-foreground">{title}</div>
      <div className="mt-1 text-foreground/80">{children}</div>
    </aside>
  );
}

const DATA_SOURCES = [
  {
    source: "Kaggle March Machine Learning Mania",
    use: "Regular-season results and box scores, seeds, men's Massey ordinals, and tournament results for the Kaggle backtests. Snapshot taken 2026-04-05.",
    cutoff: "Regular season DayNum ≤ 132, ordinals RankingDayNum ≤ 133. Tournaments start on day 134 (men) and 135 (women).",
  },
  {
    source: "College Basketball Data API (CBBD)",
    use: "Team-season game and player statistics plus tournament wins for the Ridge/SHAP bracket model, and the held-out games for the Bayesian evaluation.",
    cutoff: "Bayesian evaluation trains only on games that start strictly before the first NCAA tipoff. Player season totals have no per-game timestamps, so they rely on a declared pre-tournament snapshot.",
  },
  {
    source: "Bart Torvik T-Rank",
    use: "Adjusted efficiency (adjem) as one external forecast in the allyears EXT historical blend (2016–2024, the years without market files).",
    cutoff: "Only Time Machine exports dated Selection Sunday. A Kaggle Torvik mirror was rejected: its 2015 file starts Kentucky 38–1, which includes postseason games.",
  },
  {
    source: "Betting markets",
    use: "goto_conversion pre-deadline odds for 2025 and 2026 and archived championship futures for 2021–2024. Kalshi game markets are shown in the bracket UI but never feed a model.",
    cutoff: "Market files were published before each deadline (2025-03-18, 2026-03-17). Futures pages were archived before the tournament began.",
  },
];

export default function WriteupPage() {
  return (
    <div className="min-h-svh bg-background">
      <header className="sticky top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-3 px-4">
          <Link href="/projects/mmp" className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
            <ArrowLeft aria-hidden="true" className="size-4" />
            Back to the bracket
          </Link>
          <ThemeToggleButton />
        </div>
      </header>

      <div className="mx-auto grid max-w-6xl gap-10 px-4 pb-24 pt-12 grid-cols-[minmax(0,1fr)] lg:grid-cols-[11rem_minmax(0,1fr)]">
        <nav aria-label="On this page" className="hidden lg:block">
          <div className="sticky top-24 space-y-1 text-sm">
            <div className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">On this page</div>
            {SECTIONS.map(section => (
              <a key={section.id} href={`#${section.id}`} className="block rounded-md px-2 py-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">{section.label}</a>
            ))}
          </div>
        </nav>

        <main id="main-content" className="mx-auto w-full min-w-0 max-w-3xl">
          <div className="mb-14">
            <p className="font-mono text-xs uppercase tracking-widest text-accent">Write-up · Sports Analytics Club at UTD</p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">How the March Madness Predictor works</h1>
            <p className="mt-5 text-lg leading-8 text-muted-foreground">
              Which data we used, which models we picked and why, how we scored them against the real 2025 and 2026 tournaments, and the ideas that didn&apos;t make it.
            </p>
            <div className="mt-8 grid gap-3 sm:grid-cols-3">
              <Stat label="2025 Kaggle Brier" value="0.10797" detail="Rank #10. The top-20 cutoff was 0.10921." />
              <Stat label="2026 Kaggle Brier" value="0.12122" detail="Rank #31. The top-100 cutoff was 0.12430." />
              <Stat label="Scored games per year" value="126" detail="63 men's and 63 women's main-bracket games. First Four excluded." />
            </div>
          </div>

          <div className="space-y-14">
            <Section id="overview" eyebrow="01" title="Two problems, two models">
              <p>
                The project answers two different questions, and each one gets its own model.
              </p>
              <ul className="list-disc space-y-2 pl-5 marker:text-accent">
                <li>
                  <strong className="text-foreground">How far will each team go?</strong> The bracket on this site comes from a Ridge regression that predicts a continuous tournament-win score for every team. SHAP explains each prediction, and a Monte Carlo step turns the scores into simulated outcomes that are exported as versioned JSON.
                </li>
                <li>
                  <strong className="text-foreground">What is the probability that team A beats team B?</strong> Kaggle&apos;s March Machine Learning Mania asks for a probability for every possible pair. Only the games actually played are scored, using the Brier score: the mean squared error between the probability and the 0/1 result. For that we use a logistic regression and XGBoost ensemble blended with betting-market odds.
                </li>
              </ul>
              <p>
                Both pipelines follow the same rule. A prediction for season <em>Y</em> may only use information that existed before season <em>Y</em>&apos;s tournament, and every model is trained only on earlier seasons.
              </p>
            </Section>

            <Section id="data" eyebrow="02" title="Data sources and leak-free cutoffs">
              <p>
                Most of the work on data went into proving that nothing from after the deadline leaks in. Each source has an explicit cutoff:
              </p>
              <div className="not-prose overflow-x-auto rounded-xl border border-border bg-card shadow-sm">
                <table className="w-full min-w-[40rem] border-collapse text-left text-sm">
                  <thead className="bg-muted/60 text-xs text-muted-foreground">
                    <tr>
                      <th scope="col" className="px-4 py-2.5 font-medium">Source</th>
                      <th scope="col" className="px-4 py-2.5 font-medium">Used for</th>
                      <th scope="col" className="px-4 py-2.5 font-medium">Cutoff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {DATA_SOURCES.map(row => (
                      <tr key={row.source} className="border-t border-border align-top">
                        <th scope="row" className="px-4 py-3 font-medium text-card-foreground">{row.source}</th>
                        <td className="px-4 py-3 text-card-foreground/85">{row.use}</td>
                        <td className="px-4 py-3 text-card-foreground/85">{row.cutoff}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>
                In the Kaggle files the cutoff is a day number, so every feature module filters on it before computing anything:
              </p>
              <CodeBlock
                file="kaggle_tuning/features.py"
                code={`
"""Leak-free team-season features for March Machine Learning Mania backtests.

Everything here is computed from regular-season data (DayNum <= 132) plus seeds and
pre-tournament Massey ordinals, so features for season Y never see Y's tournament.
"""

def massey() -> pd.DataFrame:
    m = read("MMasseyOrdinals")
    m = m[m.RankingDayNum <= 133]
    last = m.sort_values("RankingDayNum").groupby(["Season", "SystemName", "TeamID"]).OrdinalRank.last().reset_index()
`}
                caption="Excerpt. The rest of massey() pivots the latest pre-tournament rank per system into -log(rank) strengths."
              />
              <p>The leakage audit (PR #66) also found traps that are easy to miss:</p>
              <ul className="list-disc space-y-1.5 pl-5 marker:text-accent">
                <li>Conference tournament games are already in the regular-season files (days 115–132), so adding them again double-counts them.</li>
                <li><C>MTeamCoaches</C> rows run to day 154, and NIT results are played during the NCAA tournament. Neither is safe to use as a feature.</li>
                <li>Women&apos;s tournament <C>WLoc</C> is recorded after the fact, because top seeds host early rounds. The 2026 solution file has no venue at all.</li>
                <li>Massey ordinals exist only for men, so the women&apos;s models have no ordinal features.</li>
              </ul>
            </Section>

            <Section id="bracket-model" eyebrow="03" title="The bracket model: Ridge + SHAP + Monte Carlo">
              <p>
                Each team-season row combines CBBD game statistics (means and standard deviations), starter and roster player aggregates, and the team&apos;s tournament wins as the target. That gives many correlated columns for only about 68 tournament teams per season.
              </p>
              <p>
                Ridge regression fits this shape well. The L2 penalty shrinks correlated features together instead of choosing one at random, the model stays linear, and a linear model gets exact SHAP attributions from <C>shap.LinearExplainer</C>. Each prediction can then be explained feature by feature. The penalty is tuned with Optuna over α ∈ [10⁻⁴, 10⁴] in 20 trials of 3-fold cross-validation inside the training seasons.
              </p>
              <p>
                Training uses an expanding window: to validate season <em>Y</em>, the model trains on every earlier season, with older seasons down-weighted:
              </p>
              <CodeBlock
                file="models/ridge_shap.py"
                code={`
def compute_recency_weights(seasons_series, target_season, half_life=3.0):
    """Calculates exponential decay weights normalized to mean 1.0."""
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    normalized_weights = weights / weights.mean()
    return normalized_weights.values
`}
                caption="Logging lines omitted. A season's weight halves for every three years between it and the target season."
              />
              <p>
                A point prediction says nothing about uncertainty, so the Monte Carlo step adds held-out residuals back onto each team&apos;s baseline. The residual pool comes from walk-forward validation seasons, never from the model&apos;s own training data:
              </p>
              <CodeBlock
                file="monte_carlo.py"
                code={`
def _bootstrap_summary(baseline, residuals, n_simulations, seed):
    """Add residuals resampled with replacement to a baseline and summarize."""
    rng = np.random.default_rng(seed)
    values = baseline + rng.choice(residuals, size=n_simulations, replace=True)
    return {
        "baseline_prediction": baseline,
        "simulated_values": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "percentile_5": float(np.percentile(values, 5)),
        "percentile_95": float(np.percentile(values, 95)),
        "probabilities": {wins: float(np.mean(values >= wins)) for wins in range(1, 7)},
    }
`}
                caption="10,000 simulations with seed 42 by default. probabilities[k] is the share of simulations reaching at least k wins."
              />
              <Callout title="What these numbers are not">
                The simulated values are success scores, continuous and unbounded. They are not calibrated bracket probabilities, and SHAP plays no part in computing them. Player season totals in the CBBD feature builder have no tournament cutoff, which is the main provenance gap in this pipeline.
              </Callout>
            </Section>

            <Section id="kaggle-model" eyebrow="04" title="The Kaggle model: logistic + XGBoost, blended with the market">
              <p>
                For pairwise probabilities the base ensemble has two members per gender, each fed differences between the two teams&apos; features:
              </p>
              <ul className="list-disc space-y-1.5 pl-5 marker:text-accent">
                <li><strong className="text-foreground">Logistic regression</strong> (C = 0.1) on a few strong ratings: seed, a ridge margin rating (&ldquo;quality&rdquo;), Elo and net efficiency, plus Massey median and Pomeroy ranks for men.</li>
                <li><strong className="text-foreground">XGBoost</strong> (max depth 2) that predicts the score margin from the same ratings plus box-score stats such as efficiency, eFG%, turnover and rebound rates, and free-throw rate. A spline fitted on out-of-fold margins turns the margin into a probability.</li>
              </ul>
              <p>
                The logistic model is hard to overfit, and the shallow trees pick up interactions it misses. Members are averaged in log-odds rather than probability space, which keeps confident predictions from being pulled toward 0.5.
              </p>
              <CodeBlock
                file="kaggle_tuning/configs/final.py"
                code={`
BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
Q = {"q_clip_M": 60, "q_clip_W": 60}
L = {"M": M0, "W": W0, "model": "logit", "C": 0.1, **Q}
X = {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **Q, "xgb": {"max_depth": 2}}
ENS = {"models": [(L, 1), (X, 1)], **Q}
BASE = {**ENS, "market_w": {"M": 0.85, "W": 0.5}, "mkt_sharpen": 1.2}
`}
                caption="The original ensemble. Later configs extend BASE rather than editing it."
              />

              <H3>Why blend in the betting market</H3>
              <p>
                Pre-deadline betting odds already include information that box scores don&apos;t, such as injuries and lineup news. Blending them in was the largest single improvement:
              </p>
              <div className="not-prose grid gap-3 sm:grid-cols-2">
                <Stat label="2025: model only → blended" value={`${marketLift.modelOnly[2025].toFixed(4)} → ${marketLift.blended[2025].toFixed(4)}`} detail="−0.0072 Brier from the market blend" />
                <Stat label="2026: model only → blended" value={`${marketLift.modelOnly[2026].toFixed(4)} → ${marketLift.blended[2026].toFixed(4)}`} detail="−0.0019 Brier from the market blend" />
              </div>
              <p>
                The blend happens after the model is calibrated. The market file is clipped to roughly [0.025, 0.975], so its log odds are sharpened by 1.2 to undo part of that clipping:
              </p>
              <CodeBlock
                file="kaggle_tuning/backtest.py"
                code={`
w = cfg.get("market_w", {}).get(g, 0.0)
if w and year in MK.PATHS:
    mk = test_pairs(g, year)[["T1", "T2"]].merge(MK.market(year), on=["T1", "T2"], how="left").mkt.values
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / np.clip(1 - x, 1e-4, 1))
    # market file is clipped to ~[0.025, 0.975]; mkt_sharpen undoes some of that
    p = 1 / (1 + np.exp(-((1 - w) * lg(p) + w * cfg.get("mkt_sharpen", 1.0) * lg(mk))))
`}
                caption="Excerpt from _post(). g is the gender, M or W."
              />
              <p>
                Men get 85% market weight and women get 50%. Market data exists only for 2025 and 2026, the two years being scored, so tuning this weight on them would be fitting to the test set. The chart below shows how flat the trade-off is:
              </p>
              <MarketSensitivityFigure />
              <p>
                The women&apos;s weight was checked the same way. Weights of 0.4, 0.5 and 0.6 give two-year means of 0.11484, 0.11490 and 0.11503. 0.4 is slightly better, but no historical market data supports choosing it, so 0.5 stayed.
              </p>

              <H3>What the tuning added</H3>
              <ul className="list-disc space-y-1.5 pl-5 marker:text-accent">
                <li><strong className="text-foreground">Men:</strong> a <C>pool_worst_conf</C> feature set. Tournament teams and teams that beat them get their own quality ratings while everyone else shares a pooled one, and it adds the worst of three Massey ranks and conference strength. Eight of 28 candidate ablations passed the acceptance rule.</li>
                <li><strong className="text-foreground">Women:</strong> 40% logistic with a ridge quality rating at α 0.1, 10% XGBoost, 30% of the original feature logistic and 20% of a port of the public &ldquo;Vilnius&rdquo; XGBoost recipe, sharpened by 1.1 in log odds.</li>
              </ul>
              <p>
                Together these form <C>combined.py:BEST</C>, which scored 0.10797 (#10) in 2025 and 0.12122 (#31) in 2026, against 0.10992 (#90) and 0.12267 (#56) for the original ensemble.
              </p>
            </Section>

            <Section id="evaluation" eyebrow="05" title="Evaluation: walk-forward Brier against the real leaderboard">
              <p>
                Every score on this page is walk-forward: the prediction for season <em>Y</em> comes from a model trained only on seasons before <em>Y</em>. The scorer was validated first. Scoring the public goto_conversion submission reproduces its leaderboard results exactly: 0.12584 in 2026 and 0.10921 in 2025.
              </p>
              <LeaderboardFigure />
              <p>
                No configuration reached the top 20 in both years. The 2026 top-20 cutoff (0.12016) stayed out of reach for every configuration reported.
              </p>

              <H3>Why tiny gains need a guard</H3>
              <p>
                Tournament outcomes are noisy. Even a seed-only model moves by about 0.02 between seasons, and a single season&apos;s 95% interval spans about ±0.03:
              </p>
              <SeedNoiseFigure />
              <p>
                So a candidate change counts only if it improves the target years and does not worsen the 2015–2024 historical mean (or the 2021–24 check) by more than <strong className="text-foreground">0.0005</strong>, compared on the same games. The guard is a robustness check, not an untouched holdout, because the target seasons were visible during selection.
              </p>
            </Section>

            <Section id="dead-ends" eyebrow="06" title="What didn't work">
              <H3>Bayesian ratings: kept opt-in</H3>
              <p>
                PR #63 fit Bayesian team ratings from regular-season wins and losses before each tournament cutoff, using 4,000 posterior draws and a Laplace approximation. It was scored on held-out NCAA games against two baselines fitted only on earlier tournaments: a seed model and a single-member Ridge SHAP + seed matchup model.
              </p>
              <BayesianFigure />
              <p>
                Bayesian ratings have a principled uncertainty story, but they lost to both baselines. They stay behind an explicit <C>--stages bayesian_monte_carlo</C> flag, and <C>run_pipeline.py --all</C> is unchanged. The models also use different inputs (wins and losses only, against seeds plus SHAP-weighted statistics), so the gap can&apos;t be blamed on Bayesian inference alone.
              </p>

              <H3>Reinforcement learning: the wrong tool</H3>
              <p>
                PR #66 checked whether RL could lower the Brier score. The answer is no, because the task lacks the structure RL depends on:
              </p>
              <div className="not-prose overflow-x-auto rounded-xl border border-border bg-card shadow-sm">
                <table className="w-full min-w-[34rem] border-collapse text-left text-sm">
                  <thead className="bg-muted/60 text-xs text-muted-foreground">
                    <tr><th scope="col" className="px-4 py-2.5 font-medium">RL needs</th><th scope="col" className="px-4 py-2.5 font-medium">The Kaggle task has</th></tr>
                  </thead>
                  <tbody className="text-card-foreground/85">
                    <tr className="border-t border-border"><td className="px-4 py-2.5">A sequence of decisions</td><td className="px-4 py-2.5">One submission before tip-off</td></tr>
                    <tr className="border-t border-border"><td className="px-4 py-2.5">Actions that change the state</td><td className="px-4 py-2.5">Games end the same way whatever is submitted</td></tr>
                    <tr className="border-t border-border"><td className="px-4 py-2.5">Partial, delayed reward (bandits)</td><td className="px-4 py-2.5">Every scored outcome revealed</td></tr>
                    <tr className="border-t border-border"><td className="px-4 py-2.5">A policy to search for</td><td className="px-4 py-2.5">A strictly proper score: the best report is the honest probability</td></tr>
                  </tbody>
                </table>
              </div>
              <p>
                The one related idea that fits is Hedge-style expert weighting with full feedback, a smoothed version of the walk-forward blend-weight choice already in use. A pre-registered protocol for testing it is written up but has not been run, so it has no measured benefit yet.
              </p>

              <H3>Other dead ends</H3>
              <ul className="list-disc space-y-1.5 pl-5 marker:text-accent">
                <li><strong className="text-foreground">Predicting bust years.</strong> None of 23 pre-tournament features (rating spread, upset rates, bid stealers and others) survived Holm correction. The strongest, men&apos;s rating spread, had ρ = −0.35 (adjusted p = 0.54).</li>
                <li><strong className="text-foreground">Lasso and Bradley–Terry ratings</strong> failed the historical guard. For example, Bradley–Terry replacements cost +0.0019 for men and +0.0085 for women.</li>
                <li><strong className="text-foreground">Isotonic calibration, stronger sharpening and full replacement of the women&apos;s model</strong> all failed the guard. A fixed 1.2 sharpening for women cost about 0.0009 historically.</li>
                <li><strong className="text-foreground">The <C>modeh</C> Vilnius variant</strong> improved 2021–24 but worsened the target-year mean, so it was rejected.</li>
                <li><strong className="text-foreground">Leaderboard gambles</strong> like the Drake pick in the 2025 market file can raise the chance of a top finish while making expected Brier worse. We didn&apos;t use any.</li>
              </ul>
            </Section>

            <Section id="reproduce" eyebrow="07" title="Reproduce it">
              <p>
                The data is gitignored. Point <C>MANIA_DATA</C> at a local mirror of the competition CSVs, then:
              </p>
              <CodeBlock
                language="bash"
                file="kaggle-score-tuning branch"
                code={`
export MANIA_DATA=/abs/path/to/data/kaggle/mirror26
export GRID_JOBS=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv sync --extra training
# model-only history, 2021-24 check, and the 2025/2026 targets
uv run python -m kaggle_tuning.grid kaggle_tuning/configs/final.py
# official-metric scores and ranks for 2021-2026
uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/mens_allyears.py:RECOMMENDED
`}
              />
              <CodeBlock
                language="bash"
                file="main branch"
                code={`
uv run --extra training python models/ridge_shap.py
uv run --extra training python run_monte_carlo.py --season 2025 \\
  --bundle model_outputs/ridge/Train_2005_to_2024_Val_2025/ridge_bundle.joblib \\
  --n-simulations 10000 --seed 42 --expected-teams 68
`}
                caption="Training the Ridge bundles, then simulating every 2025 team without retraining."
              />
              <p className="text-sm text-muted-foreground">
                Sources: the <C>main</C> README; <C>kaggle_tuning/README.md</C> and its research notes on <C>kaggle-score-tuning</C>; the Bayesian evaluation in PR #63; the Kaggle EDA and RL audit in PR #66.
              </p>
            </Section>
          </div>
        </main>
      </div>
    </div>
  );
}
