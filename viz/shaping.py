"""Pure data-shaping helpers for the plotting scripts (no matplotlib, so tests run without it)."""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLDS = list(range(1, 7))
ROUND_LABELS = ["R32", "Sweet 16", "Elite 8", "Final 4", "Title game", "Champion"]
# run_monte_carlo / shap_monte_carlo column families, best first (same order as shap_monte_carlo._best_prob_cols)
PROB_PREFIXES = {
    "tree_prob_ge_": "bracket-tree simulation",
    "calibrated_prob_ge_": "isotonic-calibrated frequency",
    "probability_ge_": "simulated frequency",
}


def prob_col(prefix: str, k: int) -> str:
    return f"{prefix}{k}_win{'s' if k != 1 else ''}"


def advancement_matrix(table: pd.DataFrame) -> tuple[np.ndarray, str]:
    """P(>= k wins) for k = 1..6 as an (n, 6) array, from the best probability columns present."""
    for prefix, label in PROB_PREFIXES.items():
        cols = [prob_col(prefix, k) for k in THRESHOLDS]
        if set(cols) <= set(table.columns) and table[cols].notna().all().all():
            return table[cols].to_numpy(float), label
    raise ValueError("Table has no complete probability_ge_<k>_win(s) column family.")


def exact_win_probs(adv: np.ndarray) -> np.ndarray:
    """(n, 6) P(>= k wins) -> (n, 7) P(exactly k wins) for k = 0..6; rows sum to 1."""
    adv = np.asarray(adv, float)
    n = len(adv)
    ge = np.column_stack([np.ones(n), adv, np.zeros(n)])
    out = ge[:, :-1] - ge[:, 1:]
    if (out < -1e-9).any() or (adv < 0).any() or (adv > 1).any():
        raise ValueError("P(>= k wins) must lie in [0, 1] and be nonincreasing in k.")
    return np.clip(out, 0.0, 1.0)


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins. Each game is counted from both sides (p, y) and (1-p, 1-y),
    so the curve does not depend on which team is listed first."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    p, y = np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    d = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(mean_p=("p", "mean"), frac=("y", "mean"), n=("y", "size"))
    d["lo"], d["hi"] = edges[d.index], edges[d.index + 1]
    return d.reset_index(drop=True)[["lo", "hi", "mean_p", "frac", "n"]]


def brier_by_season(games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """games: one row per scored game with Season, g ('M'/'W'), y and prediction columns.
    Returns Season, scope ('Combined'/'Men'/'Women'), series, brier, n. Combined pools both genders'
    games, as the Kaggle metric does. Rows where a series has no prediction are skipped."""
    rows = []
    for col in cols:
        d = games[games[col].notna()]
        sq = (d[col] - d.y) ** 2
        for scope, mask in (("Combined", slice(None)), ("Men", d.g == "M"), ("Women", d.g == "W")):
            s = sq[mask].groupby(d.Season[mask]).agg(["mean", "size"])
            rows += [{"Season": int(season), "scope": scope, "series": col, "brier": r["mean"], "n": int(r["size"])}
                     for season, r in s.iterrows()]
    return pd.DataFrame(rows, columns=["Season", "scope", "series", "brier", "n"])


def bracket_tree(slots: pd.DataFrame, seeds: pd.DataFrame, field: set[int]):
    """One season's Kaggle Slots + Seeds -> (teams, round1_pairs, tree_edges) for monte_carlo.simulate_bracket_tree.

    Play-in slots (e.g. 'W16' fed by W16a/W16b) resolve to whichever of the two teams is in `field`,
    i.e. the First Four winner: pools lock after the First Four, and the Kaggle metric excludes it.
    Games are ordered round by round, so every game's feeders come before it.
    """
    seed_team = dict(zip(seeds.Seed, seeds.TeamID))
    slot_rows = {r.Slot: (r.StrongSeed, r.WeakSeed) for r in slots.itertuples()}

    def resolve(label: str) -> int:
        if label in seed_team:
            return int(seed_team[label])
        if label not in slot_rows:
            raise ValueError(f"Unknown seed or slot {label!r}.")
        alive = [seed_team[s] for s in slot_rows[label] if seed_team.get(s) in field]
        if len(alive) != 1:
            raise ValueError(f"Play-in slot {label!r} must have exactly one team in the field, found {len(alive)}.")
        return int(alive[0])

    rounds = slots[slots.Slot.str.match(r"^R\d")].assign(rnd=lambda d: d.Slot.str[1].astype(int))
    rounds = rounds.sort_values(["rnd", "Slot"])
    teams, pairs, edges, game = [], [], [], {}
    for r in rounds[rounds.rnd == 1].itertuples():
        teams += [resolve(r.StrongSeed), resolve(r.WeakSeed)]
        game[r.Slot] = len(pairs)
        pairs.append((len(teams) - 2, len(teams) - 1))
    for r in rounds[rounds.rnd > 1].itertuples():
        game[r.Slot] = len(pairs) + len(edges)
        edges.append((game[r.StrongSeed], game[r.WeakSeed]))
    if len(set(teams)) != len(teams):
        raise ValueError("A team appears in more than one first-round game.")
    return teams, pairs, edges


def actual_wins(played: pd.DataFrame) -> pd.Series:
    """played: T1, T2, y (1 if T1 won). Returns wins per TeamID, counting every team that played."""
    winners = np.where(played.y.values == 1, played.T1.values, played.T2.values)
    teams = pd.Index(np.unique(np.concatenate([played.T1.values, played.T2.values])))
    return pd.Series(winners).value_counts().reindex(teams, fill_value=0).astype(int)
