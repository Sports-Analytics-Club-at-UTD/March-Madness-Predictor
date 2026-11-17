// Static numbers for the write-up page, copied from the research notes in the
// Python repo. Each block names its source; nothing here is computed on the site.

/** Leaderboard cutoffs (kaggle_tuning/README.md on origin/kaggle-score-tuning). */
export const leaderboardCutoffs = {
  2025: { top100: 0.11078, top20: 0.10921 },
  2026: { top100: 0.1243, top20: 0.12016 },
} as const;

export type ScoredConfig = {
  key: string;
  label: string;
  note: string;
  scores: { 2025: number; 2026: number };
  ranks?: { 2025: number; 2026: number };
};

/**
 * Combined men's + women's Brier on the 126 main-bracket games.
 * Sources: kaggle_tuning/README.md (market-only file), configs/final.py docstring and
 * vilnius_results.md (final), MENS_ALLYEARS.md (combined.py:BEST), and
 * research/ratings_results/REPORT.md (Efficiency candidate).
 */
export const leaderboardConfigs: ScoredConfig[] = [
  { key: "market", label: "Betting odds only", note: "goto_conversion file; 2025 includes its Drake gamble", scores: { 2025: 0.10921, 2026: 0.12584 } },
  { key: "final", label: "Original ensemble + market", note: "configs/final.py", scores: { 2025: 0.10992, 2026: 0.12267 }, ranks: { 2025: 90, 2026: 56 } },
  { key: "best", label: "Tuned ensemble (combined BEST)", note: "configs/combined.py:BEST", scores: { 2025: 0.107965, 2026: 0.121219 }, ranks: { 2025: 10, 2026: 31 } },
  { key: "efficiency", label: "+ ridge efficiency ratings", note: "ratings.py:EFFICIENCY, exploratory", scores: { 2025: 0.1077, 2026: 0.12103 }, ranks: { 2025: 9, 2026: 31 } },
];

/** Model-only vs market-blended `final` (vilnius_results.md). */
export const marketLift = {
  modelOnly: { 2025: 0.11715, 2026: 0.12457 },
  blended: { 2025: 0.10992, 2026: 0.12267 },
};

/** Men's market-weight sensitivity at market sharpening 1.2 (MENS_RESEARCH.md). */
export const mensMarketSensitivity = [
  { weight: 70, m25: 0.125578, m26: 0.144849, mean: 0.135213 },
  { weight: 75, m25: 0.125607, m26: 0.144728, mean: 0.135168 },
  { weight: 80, m25: 0.125679, m26: 0.144623, mean: 0.135151 },
  { weight: 85, m25: 0.125793, m26: 0.144533, mean: 0.135163 },
  { weight: 90, m25: 0.125946, m26: 0.144458, mean: 0.135202 },
  { weight: 95, m25: 0.126136, m26: 0.144398, mean: 0.135267 },
  { weight: 100, m25: 0.12636, m26: 0.144353, mean: 0.135357 },
];

/** Seed-only walk-forward reference Brier (research/KAGGLE_EDA.md on PR #66). */
export const seedOnlyBySeason = [
  { season: 2015, men: 0.1704, women: 0.147 },
  { season: 2016, men: 0.2028, women: 0.1719 },
  { season: 2017, men: 0.1765, women: 0.1408 },
  { season: 2018, men: 0.2063, women: 0.1676 },
  { season: 2019, men: 0.1735, women: 0.1313 },
  { season: 2021, men: 0.2291, women: 0.1428 },
  { season: 2022, men: 0.219, women: 0.172 },
  { season: 2023, men: 0.2047, women: 0.1695 },
  { season: 2024, men: 0.1915, women: 0.1132 },
  { season: 2025, men: 0.1528, women: 0.1121 },
  { season: 2026, men: 0.1653, women: 0.117 },
];

/** Expanded Bayesian evaluation, 801 games (docs/bayesian-evaluation-expanded.md on PR #63). */
export const bayesianBySeason = [
  { season: 2010, bayesian: 0.1911, seed: 0.2122, matchup: 0.2017 },
  { season: 2011, bayesian: 0.2256, seed: 0.214, matchup: 0.2026 },
  { season: 2012, bayesian: 0.2085, seed: 0.1967, matchup: 0.1886 },
  { season: 2013, bayesian: 0.2083, seed: 0.213, matchup: 0.1899 },
  { season: 2014, bayesian: 0.2188, seed: 0.2188, matchup: 0.2141 },
  { season: 2015, bayesian: 0.1941, seed: 0.178, matchup: 0.1625 },
  { season: 2016, bayesian: 0.2111, seed: 0.2043, matchup: 0.1963 },
  { season: 2019, bayesian: 0.1921, seed: 0.1791, matchup: 0.1876 },
  { season: 2022, bayesian: 0.2288, seed: 0.2189, matchup: 0.2024 },
  { season: 2023, bayesian: 0.2394, seed: 0.2069, matchup: 0.2111 },
  { season: 2024, bayesian: 0.2139, seed: 0.1946, matchup: 0.1869 },
  { season: 2025, bayesian: 0.1733, seed: 0.1612, matchup: 0.1583 },
];

export const bayesianPooled = [
  { key: "bayesian", label: "Bayesian ratings", brier: 0.2088, logLoss: 0.6011, accuracy: 0.6492 },
  { key: "seed", label: "Seeds only", brier: 0.1998, logLoss: 0.5821, accuracy: 0.6804 },
  { key: "matchup", label: "Ridge SHAP + seed", brier: 0.1918, logLoss: 0.5642, accuracy: 0.7079 },
];
