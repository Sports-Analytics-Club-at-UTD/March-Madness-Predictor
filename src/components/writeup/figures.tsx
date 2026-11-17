"use client";

import { ChartFrame, DataTable, DotPlot, LineChart, type SeriesColor } from "@/components/writeup/charts";
import { bayesianBySeason, bayesianPooled, leaderboardConfigs, leaderboardCutoffs, mensMarketSensitivity, seedOnlyBySeason } from "@/lib/writeup-data";

const CONFIG_COLOR: Record<string, SeriesColor> = { market: "reference", final: "accent", best: "primary", efficiency: "primary" };
const YEARS = [2025, 2026] as const;

export function LeaderboardFigure() {
  return (
    <ChartFrame
      title="Combined Brier on the real 2025 and 2026 tournaments"
      subtitle="Lower is better. Dashed lines are the final Kaggle leaderboard cutoffs for the top 20 and top 100. Hover a dot for its leaderboard rank."
      table={<DataTable columns={["Configuration", "2025", "Rank", "2026", "Rank"]} rows={leaderboardConfigs.map(c => [c.label, c.scores[2025].toFixed(5), c.ranks ? `#${c.ranks[2025]}` : "n/a", c.scores[2026].toFixed(5), c.ranks ? `#${c.ranks[2026]}` : "n/a"])} />}
      footnote="Betting odds only is the public goto_conversion file. Its 2025 score includes a deliberate Drake gamble. The ridge efficiency row is an exploratory candidate. Neither 2025 nor 2026 was an untouched holdout, because both were inspected while configurations were being chosen."
    >
      <div className="space-y-5">
        {YEARS.map(year => (
          <div key={year}>
            <div className="mb-1 font-mono text-xs font-medium text-muted-foreground">{year}</div>
            <DotPlot
              ariaLabel={`Brier score by configuration for ${year}`}
              rows={leaderboardConfigs.map(c => ({
                key: c.key,
                label: c.label,
                value: c.scores[year],
                color: CONFIG_COLOR[c.key],
                detail: c.ranks ? `Leaderboard rank #${c.ranks[year]}` : c.note,
              }))}
              references={[
                { value: leaderboardCutoffs[year].top20, label: "top 20" },
                { value: leaderboardCutoffs[year].top100, label: "top 100" },
              ]}
              format={value => value.toFixed(5)}
            />
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}

const SENSITIVITY_PANELS = [
  { key: "m25", title: "Men 2025" },
  { key: "m26", title: "Men 2026" },
  { key: "mean", title: "Two-year mean" },
] as const;

export function MarketSensitivityFigure() {
  const categories = mensMarketSensitivity.map(row => `${row.weight}%`);
  const chosen = mensMarketSensitivity.findIndex(row => row.weight === 85);
  return (
    <ChartFrame
      title="How much weight should the men's market odds get?"
      subtitle="Men's Brier as the market's share of the log-odds blend changes (market sharpening fixed at 1.2). Each panel has its own scale, and the large dot marks that panel's minimum."
      table={<DataTable columns={["Market weight", "Men 2025", "Men 2026", "Mean"]} rows={mensMarketSensitivity.map(row => [`${row.weight}%`, row.m25.toFixed(6), row.m26.toFixed(6), row.mean.toFixed(6)])} />}
      footnote="In 2025 less market weight is better and in 2026 more is better. The two-year optimum of 80% beats 85% by only 0.000012. With no pre-2025 market data available to tune against, the established 85% was kept."
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {SENSITIVITY_PANELS.map(panel => (
          <div key={panel.key}>
            <div className="mb-1 text-xs font-medium text-muted-foreground">{panel.title}</div>
            <LineChart
              ariaLabel={`${panel.title} Brier by market weight`}
              categories={categories}
              series={[{ key: panel.key, label: panel.title, color: panel.key === "mean" ? "primary" : "accent", values: mensMarketSensitivity.map(row => row[panel.key]) }]}
              markers={[{ index: chosen, label: "85%" }]}
              highlightMin
              height={180}
              format={value => value.toFixed(6)}
            />
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}

export function SeedNoiseFigure() {
  return (
    <ChartFrame
      title="A seed-only model already swings by ±0.03 between seasons"
      subtitle="Walk-forward Brier of P(win) = sigmoid(b · seed gap), fitted only on earlier seasons. 2020 had no tournament."
      table={<DataTable columns={["Season", "Men", "Women"]} rows={seedOnlyBySeason.map(row => [row.season, row.men.toFixed(4), row.women.toFixed(4)])} />}
      footnote="Mean 0.1902 for men and 0.1441 for women (SD 0.024 each). The tuning gains on this page are measured in ten-thousandths, so only paired comparisons on the same games can rank configurations."
    >
      <LineChart
        ariaLabel="Seed-only Brier by season for men and women"
        categories={seedOnlyBySeason.map(row => row.season)}
        series={[
          { key: "men", label: "Men", color: "accent", values: seedOnlyBySeason.map(row => row.men) },
          { key: "women", label: "Women", color: "primary", values: seedOnlyBySeason.map(row => row.women) },
        ]}
        height={250}
      />
    </ChartFrame>
  );
}

const BAYES_COLOR: Record<string, SeriesColor> = { bayesian: "accent", seed: "reference", matchup: "primary" };

export function BayesianFigure() {
  return (
    <ChartFrame
      title="Bayesian ratings lost to both baselines"
      subtitle="Held-out NCAA game probabilities over 12 seasons and 801 identical games, First Four included. Lower Brier is better."
      table={<DataTable columns={["Season", "Bayesian", "Seeds", "Ridge SHAP + seed"]} rows={bayesianBySeason.map(row => [row.season, row.bayesian.toFixed(4), row.seed.toFixed(4), row.matchup.toFixed(4)])} />}
      footnote="Bayesian beat seeds in 2 of 12 seasons and Ridge SHAP + seed in 1 of 12. Season-bootstrap 95% intervals for Bayesian minus baseline Brier are [+0.0017, +0.0157] against seeds and [+0.0100, +0.0229] against the matchup model, so both exclude zero."
    >
      <div className="space-y-6">
        <DotPlot
          ariaLabel="Pooled Brier for the Bayesian, seed and Ridge SHAP models"
          rows={bayesianPooled.map(model => ({ key: model.key, label: model.label, value: model.brier, color: BAYES_COLOR[model.key], detail: `Log loss ${model.logLoss.toFixed(4)} · accuracy ${(model.accuracy * 100).toFixed(1)}%` }))}
          domain={[0.185, 0.215]}
        />
        <LineChart
          ariaLabel="Per-season Brier for the Bayesian, seed and Ridge SHAP models"
          categories={bayesianBySeason.map(row => row.season)}
          series={bayesianPooled.map(model => ({ key: model.key, label: model.label, color: BAYES_COLOR[model.key], values: bayesianBySeason.map(row => row[model.key as "bayesian" | "seed" | "matchup"]) }))}
          height={250}
        />
      </div>
    </ChartFrame>
  );
}
