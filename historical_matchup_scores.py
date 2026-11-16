"""Export retrospective SHAP ensemble scores using strictly earlier artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bayesian_evaluation import digest, load_config, write_csv, write_json
from bayesian_ratings import _provider_id, utc_timestamp
from shap_scoring import dot_zscore


SKIP_COLUMNS = {"gameId", "season", "teamId", "opponentId", "teamSeed", "opponentSeed",
                "totalMinutes", "games", "wins", "losses", "starts"}


def numeric_columns(frame):
    """Infer numeric types after cutoff filtering; future text cannot poison dtypes."""
    result = {}
    for name in frame:
        column = frame[name]
        if pd.api.types.is_bool_dtype(column):
            continue
        converted = pd.to_numeric(column, errors="coerce")
        if converted.notna().sum() == column.notna().sum():
            result[name] = converted
    return pd.DataFrame(result, index=frame.index)


def build_cutoff_features(data_dir, spec, manifest_path):
    """Match combo_eval's feature formula, with explicit time and snapshot checks."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("season") != spec.season or manifest.get("status") != "complete":
        raise ValueError(f"{spec.season}: require a complete same-season statistics manifest")
    if manifest.get("statistics_scope") != "regular season, before NCAA tournament":
        raise ValueError(f"{spec.season}: manifest does not declare pre-tournament regular-season statistics")
    as_of = utc_timestamp(manifest["statistics_cutoff"])
    if as_of >= spec.cutoff:
        raise ValueError(f"{spec.season}: declared statistics snapshot must precede the scoring cutoff")
    paths = sorted((Path(data_dir) / str(spec.season)).glob("*/*_game_stats.csv"))
    rows, inputs, excluded = [], [Path(manifest_path)], {"cutoff": 0, "nonregular": 0, "unfinished": 0}
    latest_game = None
    for game_path in paths:
        slug = game_path.parent.name
        player_path = game_path.with_name(f"{slug}_players_season_stats.csv")
        if not player_path.exists():
            raise ValueError(f"Missing player statistics: {player_path}")
        games = pd.read_csv(game_path)
        players = pd.read_csv(player_path)
        if not {"season", "seasonType", "startDate", "teamId", "team"} <= set(games):
            raise ValueError(f"Missing identity/time columns in {game_path}")
        if not (pd.to_numeric(games["season"], errors="raise") == spec.season).all():
            raise ValueError(f"Different season in {game_path}")
        regular = games["seasonType"].astype(str).str.strip().str.lower() == "regular"
        excluded["nonregular"] += int((~regular).sum())
        games = games.loc[regular].copy()
        starts = games["startDate"].map(utc_timestamp)
        before = starts < spec.cutoff
        excluded["cutoff"] += int((~before).sum())
        games, starts = games.loc[before].copy(), starts.loc[before]
        if "status" in games:
            final = games["status"].fillna("").str.strip().str.lower().isin(["", "final", "completed"])
            excluded["unfinished"] += int((~final).sum())
            games, starts = games.loc[final].copy(), starts.loc[final]
        if games.empty:
            raise ValueError(f"No eligible regular-season features in {game_path}")
        if starts.max() > as_of:
            raise ValueError(f"{game_path}: game starts exceed the declared statistics snapshot")
        latest_game = max(latest_game, starts.max()) if latest_game is not None else starts.max()
        identities = games[["teamId", "team"]].drop_duplicates()
        if len(identities) != 1:
            raise ValueError(f"Ambiguous team identity in {game_path}")
        # An excluded blank row can make pandas infer a valid integer ID as float.
        identity_number = float(identities.iloc[0]["teamId"])
        if not np.isfinite(identity_number) or not identity_number.is_integer():
            raise ValueError(f"Invalid numeric team identity in {game_path}")
        team_id = _provider_id(str(int(identity_number)))
        if "season" not in players or not (pd.to_numeric(players["season"], errors="raise") == spec.season).all():
            raise ValueError(f"Different or absent player season in {player_path}")
        if players.empty or "minutes" not in players:
            raise ValueError(f"Missing player minutes in {player_path}")
        if "seasonType" in players and not players["seasonType"].astype(str).str.lower().eq("regular").all():
            raise ValueError(f"Player snapshot contains nonregular statistics in {player_path}")
        if "teamId" in players and not pd.to_numeric(players["teamId"], errors="raise").eq(int(team_id[5:])).all():
            raise ValueError(f"Player/team identity mismatch in {player_path}")
        player_numeric = numeric_columns(players)
        if "minutes" not in player_numeric or player_numeric["minutes"].isna().any():
            raise ValueError(f"Invalid player minutes in {player_path}")
        top5 = player_numeric.nlargest(5, "minutes")
        record = {"team_id": team_id, "team": str(identities.iloc[0]["team"]).strip()}
        for name, column in numeric_columns(games).items():
            if name not in SKIP_COLUMNS and not name.endswith("EloEnd"):
                record[f"game_mean_{name}"] = column.mean()
                record[f"game_std_{name}"] = column.std()
        for name in player_numeric:
            if name not in SKIP_COLUMNS and "Id" not in name:
                record[f"starters_mean_{name}"] = top5[name].mean()
                record[f"roster_max_{name}"] = player_numeric[name].max()
        rows.append(record)
        inputs.extend([game_path, player_path])
    features = pd.DataFrame(rows).fillna(0)
    if features.empty or features["team_id"].duplicated().any():
        raise ValueError(f"{spec.season}: need nonempty, unique cached team identities")
    matrix = features.drop(columns=["team_id", "team"])
    if matrix.empty or not np.isfinite(matrix.to_numpy(float)).all():
        raise ValueError(f"{spec.season}: need finite numeric features")
    return features, inputs, {"as_of": as_of.isoformat(), "latest_game_start": latest_game.isoformat(),
                              "excluded_rows": excluded, "normalization_teams": len(features),
                              "manifest_sha256": digest(manifest_path)}


def select_earlier_weights(training_output_dir, member, season):
    """Require both model training and SHAP validation features to be earlier."""
    if not re.fullmatch(r"[a-zA-Z0-9_]+", member):
        raise ValueError(f"Invalid member name: {member!r}")
    candidates = []
    for path in (Path(training_output_dir) / member).glob("Train_*_Val_*/shap_feature_importance.csv"):
        match = re.fullmatch(r"Train_(\d{4})_to_(\d{4})_Val_(\d{4})", path.parent.name)
        if match:
            first, last, validation = map(int, match.groups())
            if first <= last < validation < season:
                candidates.append((validation, first, last, path))
    if not candidates:
        raise ValueError(f"{member}/{season}: no strictly earlier walk-forward SHAP artifact")
    latest = max(c[0] for c in candidates)
    selected = [c for c in candidates if c[0] == latest]
    if len(selected) != 1:
        raise ValueError(f"{member}/{season}: ambiguous earlier SHAP fold")
    validation, first, last, path = selected[0]
    bundle_path = path.with_name(f"{member}_bundle.joblib")
    if not bundle_path.exists():
        raise ValueError(f"Missing inference metadata for {path}: {bundle_path}")
    bundle = joblib.load(bundle_path)
    training = bundle["training_seasons"]
    if (not training or any(isinstance(year, bool) or not isinstance(year, (int, np.integer)) for year in training)
            or min(training) != first or max(training) != last or any(year >= validation for year in training)
            or bundle["validation_season"] != validation):
        raise ValueError(f"{bundle_path}: training/validation metadata contradicts the earlier fold")
    frame = pd.read_csv(path)
    if not {"feature", "shap_importance"} <= set(frame):
        raise ValueError(f"Missing SHAP columns in {path}")
    values = pd.to_numeric(frame["shap_importance"], errors="raise")
    if (frame.empty or frame["feature"].isna().any() or frame["feature"].duplicated().any()
            or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any()
            or not set(frame["feature"]) <= set(bundle["feature_names"])):
        raise ValueError(f"Invalid SHAP weights or feature provenance in {path}")
    return dict(zip(frame["feature"], values)), {
        "member": member, "validation_season": validation, "training_seasons": [int(y) for y in training],
        "producer_through_season": validation, "weights_path": str(path.resolve()),
        "weights_sha256": digest(path), "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": digest(bundle_path),
    }


def export_historical_scores(config_path, output_dir, *, members, data_dir=Path("data"),
                              training_output_dir=Path("model_outputs"), manifest_dir=Path("data/brackets")):
    specs, _ = load_config(config_path)
    if not members or len(set(members)) != len(members):
        raise ValueError("Supply a nonempty fixed list of unique ensemble members")
    declared_members = json.loads(Path(config_path).read_text()).get("matchup_members")
    if declared_members is not None and list(members) != declared_members:
        raise ValueError("Requested members differ from the fixed membership in the configuration")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, seasons, sources = [], {}, {}
    for spec in specs:
        frame, inputs, info = build_cutoff_features(data_dir, spec, Path(manifest_dir) / str(spec.season) / "manifest.json")
        matrix = frame.drop(columns=["team_id", "team"])
        scaled = StandardScaler().fit_transform(matrix)
        member_scores, producer_through, provenance = {}, [], []
        for member in members:
            weights, artifact = select_earlier_weights(training_output_dir, member, spec.season)
            used = set(weights) & set(matrix)
            if not used or not any(weights[name] > 0 for name in used):
                raise ValueError(f"{member}/{spec.season}: no usable positive SHAP weights")
            scores, missing = dot_zscore(matrix, scaled, weights)
            if np.std(scores) < 1e-12:
                raise ValueError(f"{member}/{spec.season}: SHAP scores have no variation")
            member_scores[member] = scores
            producer_through.append(artifact["producer_through_season"])
            provenance.append(dict(artifact, missing_features=missing,
                                   missing_weight_fraction=float(sum(weights[n] for n in missing) / sum(weights.values()))))
        ensemble = np.mean(np.column_stack(list(member_scores.values())), axis=1)
        for index, row in frame.iterrows():
            rows.append({"season": spec.season, "team_id": row["team_id"], "team": row["team"],
                         "ensemble_score": float(ensemble[index]), "as_of": info["as_of"],
                         "trained_through_season": max(producer_through),
                         **{f"score__{member}": float(values[index]) for member, values in member_scores.items()}})
        seasons[str(spec.season)] = dict(info, members=provenance)
        for path in inputs:
            sources[str(path.resolve())] = digest(path)
    write_csv(output_dir / "ensemble_scores.csv", rows)
    write_json(output_dir / "provenance.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "members": list(members), "membership_selection": "fixed before evaluation; no outcome-based recipe selection",
        "comparison_label": "single-member Ridge SHAP + seed" if list(members) == ["ridge"] else "fixed-member SHAP ensemble + seed",
        "config_sha256": digest(config_path), "seasons": seasons, "inputs_sha256": sources,
        "scores_sha256": digest(output_dir / "ensemble_scores.csv"),
        "limitations": [
            "Player snapshot availability relies on the existing statistics manifest; no independent point-in-time archive was verified.",
            "This uses strictly earlier SHAP validation folds, not a full-history Rank 1 recipe or a same-season fold.",
            "Saved artifact training metadata is checked; historical data corrections and upstream tuning provenance are not independently established.",
        ],
    })
    return output_dir / "ensemble_scores.csv"
