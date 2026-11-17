"""Bracket / Monte Carlo plots.

    MANIA_DATA=/abs/path/mirror26 uv run --extra training python -m viz.bracket --season 2026
    uv run --extra training python -m viz.bracket --scores-csv model_outputs/monte_carlo/team_success_scores_shap_2025.csv

Default: runs the bracket pipeline's simulator (monte_carlo.simulate_bracket_tree) over the official
Kaggle bracket of every backtest season, with pairwise win probabilities from the walk-forward
Kaggle model (shared cache with viz.kaggle), for men and women. Writes, for --season, an
advancement heatmap and a wins distribution, plus round-advancement Brier by season against a
seed-only baseline.

--scores-csv instead plots a run_monte_carlo.py output table (team + probability_ge_<k>_win(s),
tree_/calibrated_ variants preferred): heatmap and wins distribution only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .shaping import ROUND_LABELS, THRESHOLDS, actual_wins, advancement_matrix, bracket_tree, exact_win_probs, prob_col

GENDERS = {"M": "Men", "W": "Women"}


def simulate_season(preds: pd.DataFrame, g: str, year: int, n_sims: int, seed: int) -> pd.DataFrame:
    """preds: T1, T2, p for this season/gender/variant. Returns one row per bracket team in
    run_monte_carlo's schema (team, season, probability_ge_<k>_win(s), simulated_mean) plus TeamID, seed, actual_wins."""
    from kaggle_tuning import backtest as B
    from kaggle_tuning import features as F
    from monte_carlo import simulate_bracket_tree

    seeds = F.read(f"{g}NCAATourneySeeds").query("Season == @year")
    slots = F.read(f"{g}NCAATourneySlots").query("Season == @year")
    played = B.test_pairs(g, year)
    teams, pairs, edges = bracket_tree(slots, seeds, set(played.T1) | set(played.T2))
    idx = {t: i for i, t in enumerate(teams)}
    m = np.full((len(teams), len(teams)), np.nan)
    d = preds[preds.T1.isin(idx) & preds.T2.isin(idx)]
    i, j = d.T1.map(idx).to_numpy(), d.T2.map(idx).to_numpy()
    m[i, j], m[j, i] = d.p.to_numpy(), 1 - d.p.to_numpy()
    np.fill_diagonal(m, 0.5)
    if np.isnan(m).any():
        raise ValueError(f"{g} {year}: missing pairwise probabilities for the bracket field.")
    counts, _ = simulate_bracket_tree(pairs, edges, teams, m, n_simulations=n_sims, seed=seed)
    names = F.read(f"{g}Teams").set_index("TeamID").TeamName
    out = pd.DataFrame({"team": names.reindex(teams).values, "season": year, "gender": g, "TeamID": teams,
                        "seed": seeds.set_index("TeamID").Seed.str[1:3].astype(int).reindex(teams).values,
                        "simulated_mean": counts.mean(axis=1), "actual_wins": actual_wins(teams, pairs, edges, played)})
    for k in THRESHOLDS:
        out[prob_col("probability_ge_", k)] = (counts >= k).mean(axis=1)
    return out.sort_values(["simulated_mean", "team"], ascending=[False, True]).reset_index(drop=True)


def _label(row) -> str:
    seed = getattr(row, "seed", None)
    return f"{row.team} ({int(seed)})" if seed is not None and pd.notna(seed) else str(row.team)


def plot_heatmap(panels: dict[str, pd.DataFrame], path: Path, title: str, top: int = 16) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from . import style as S

    fig, axes = plt.subplots(1, len(panels), figsize=(max(9.0, 6.6 * len(panels)), 0.42 * top + 2.2), squeeze=False,
                             gridspec_kw={"wspace": 0.45})
    for ax, (name, table) in zip(axes[0], panels.items()):
        t = table.head(top)
        adv, source = advancement_matrix(t)
        ax.imshow(adv, cmap=S.SEQ, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for r in range(adv.shape[0]):
            for c in range(adv.shape[1]):
                v = adv[r, c]
                ax.text(c, r, f"{v:.0%}" if v >= 0.01 else "<1%", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else S.INK)
            if "actual_wins" in t and pd.notna(t.actual_wins.iloc[r]) and t.actual_wins.iloc[r] >= 1:
                c = min(int(t.actual_wins.iloc[r]), 6) - 1
                ax.add_patch(Rectangle((c - 0.47, r - 0.45), 0.94, 0.9, fill=False, ec=S.ORANGE, lw=2.4))
        ax.set_xticks(range(6), [lab.replace(" ", "\n") for lab in ROUND_LABELS])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(t)), [_label(row) for row in t.itertuples()])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}  ·  {source}", loc="left", pad=40)
    note = "Probability of reaching each round, top teams by expected wins."
    if any("actual_wins" in t for t in panels.values()):
        note += "  Orange box = furthest round actually reached (none = lost first game)."
    fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.07)
    return S.save(fig, path)


def plot_wins_distribution(panels: dict[str, pd.DataFrame], path: Path, title: str, top: int = 12) -> Path:
    import matplotlib.pyplot as plt

    from . import style as S

    colors = ["#e8eef6"] + S.BLUE_RAMP[:6]  # 0 wins recedes; 1..6 wins step light -> dark
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 0.42 * top + 2.0), squeeze=False,
                             gridspec_kw={"wspace": 0.4})
    for ax, (name, table) in zip(axes[0], panels.items()):
        t = table.head(top).iloc[::-1]
        exact = exact_win_probs(advancement_matrix(t)[0])
        y = np.arange(len(t))
        left = np.zeros(len(t))
        for k in range(7):
            ax.barh(y, exact[:, k], left=left, height=0.72, color=colors[k], edgecolor=S.SURFACE, lw=1.5,
                    label="Champion (6)" if k == 6 else f"{k} win{'s' if k != 1 else ''}")
            left += exact[:, k]
        if "actual_wins" in t:
            aw = t.actual_wins.to_numpy()
            ok = ~pd.isna(aw)
            cum = np.column_stack([np.zeros(len(t)), np.cumsum(exact, axis=1)])
            k = aw[ok].astype(int)
            mid = (cum[ok, k] + cum[ok, k + 1]) / 2
            ax.scatter(mid, y[ok], marker="D", s=46, color=S.ORANGE, edgecolor=S.INK, lw=0.8, zorder=4,
                       label="Actual result")
        ax.set_yticks(y, [_label(row) for row in t.itertuples()])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of simulations")
        ax.set_title(name, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"{title}\nSimulated tournament wins for the top teams by expected wins", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.02)
    return S.save(fig, path)


def advancement_brier(tables: pd.DataFrame) -> pd.DataFrame:
    """tables: simulate_season rows with a `variant` column. Brier of P(>= k wins) vs reaching round k,
    over the 64 bracket teams, per season/gender/variant/k."""
    rows = []
    for (year, g, variant), t in tables.groupby(["season", "gender", "variant"]):
        adv = advancement_matrix(t)[0]
        hit = t.actual_wins.to_numpy()[:, None] >= np.array(THRESHOLDS)
        for k, b in zip(THRESHOLDS, ((adv - hit) ** 2).mean(axis=0)):
            rows.append({"season": year, "gender": g, "variant": variant, "k": k, "brier": b})
    return pd.DataFrame(rows)


def plot_advancement_brier(scores: pd.DataFrame, path: Path, title: str, ks=(2, 4, 6)) -> Path:
    import matplotlib.pyplot as plt

    from . import style as S

    seasons = np.arange(scores.season.min(), scores.season.max() + 1)
    fig, axes = plt.subplots(len(ks), 2, figsize=(13, 3.0 * len(ks)), sharex=True)
    for r, k in enumerate(ks):
        for c, g in enumerate(GENDERS):
            ax = axes[r, c]
            for variant, color, ls, lab in (("full", S.BLUE, "-", "Kaggle model"), ("seed", S.GRAY, "--", "Seed-only logit")):
                s = scores.query("gender == @g and variant == @variant and k == @k").set_index("season").brier.reindex(seasons)
                ax.plot(seasons, s, color=color, ls=ls, marker="o", label=lab)
            ax.set_title(f"{GENDERS[g]}: P(reach {ROUND_LABELS[k - 1]})", loc="left")
            if c == 0:
                ax.set_ylabel("Brier (64 teams)")
    for ax in axes[-1]:
        ax.set_xticks(seasons[seasons != 2020])
        ax.set_xticklabels([f"'{y % 100:02d}" for y in seasons if y != 2020])
        ax.set_xlabel("Tournament season (no 2020 tournament)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{title}\nRound-advancement Brier by season (each panel has its own scale)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return S.save(fig, path)


def main(argv=None):
    from .kaggle import DEFAULT_CONFIG, DEFAULT_OUT, field_predictions, load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026, help="season for the heatmap and wins distribution")
    ap.add_argument("--scores-csv", type=Path, help="plot a run_monte_carlo.py output table instead")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-simulations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--jobs", type=int)
    a = ap.parse_args(argv)
    out = a.out / "bracket"

    if a.scores_csv:
        table = pd.read_csv(a.scores_csv).sort_values("simulated_mean", ascending=False)
        season = int(table.season.iloc[0]) if "season" in table else a.season
        panels = {f"{season} Monte Carlo": table}
        title = f"{season} bracket pipeline: {a.scores_csv.name}"
        plot_heatmap(panels, out / f"advancement_heatmap_{a.scores_csv.stem}.png", title)
        plot_wins_distribution(panels, out / f"wins_distribution_{a.scores_csv.stem}.png", title)
        return

    name = load_config(a.config)[0]
    preds = field_predictions(a.config, out=a.out, refresh=a.refresh, jobs=a.jobs)
    tables = []
    for (year, g, variant), d in preds[preds.variant.isin(["full", "seed"])].groupby(["Season", "g", "variant"]):
        tables.append(simulate_season(d, g, int(year), a.n_simulations, a.seed).assign(variant=variant))
    tables = pd.concat(tables, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    tables.to_csv(out / f"advancement_{name}.csv", index=False)
    scores = advancement_brier(tables)
    scores.to_csv(out / "advancement_brier_by_season.csv", index=False)

    full = tables[(tables.variant == "full") & (tables.season == a.season)]
    if full.empty:
        raise SystemExit(f"No simulated bracket for {a.season}.")
    panels = {GENDERS[g]: full[full.gender == g] for g in GENDERS}
    title = f"{a.season} bracket simulation ({a.n_simulations:,} runs, Kaggle model {name})"
    plot_heatmap(panels, out / f"advancement_heatmap_{a.season}.png", title)
    plot_wins_distribution(panels, out / f"wins_distribution_{a.season}.png", title)
    plot_advancement_brier(scores, out / "advancement_brier_by_season.png",
                           f"Bracket simulation backtest (Kaggle model {name} vs seed-only)")


if __name__ == "__main__":
    main()
