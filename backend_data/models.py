"""Versioned wire models. JSON Schema is generated from these definitions."""

from collections import Counter
from datetime import datetime, timezone
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, JsonValue,
    model_validator,
)

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")]
Text = Annotated[str, Field(min_length=1)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
PositiveInt = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[FiniteFloat, Field(ge=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class Interval(ContractModel):
    """Intervals retain their statistical interpretation and target quantity."""

    kind: Literal["empirical_prediction", "confidence", "credible"]
    target: Text
    level: Annotated[FiniteFloat, Field(gt=0, lt=1)]
    lower: FiniteFloat
    upper: FiniteFloat
    method: Text

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("Interval lower bound exceeds upper bound")
        return self


class PosteriorSummary(ContractModel):
    """Optional interface for a future Bayesian producer; no posterior is inferred here."""

    target: Text
    mean: FiniteFloat
    median: FiniteFloat | None = None
    std: Nonnegative | None = None
    intervals: list[Interval] = Field(default_factory=list)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def credible_intervals(self) -> Self:
        if any(i.kind != "credible" or i.target != self.target for i in self.intervals):
            raise ValueError("Posterior intervals must be credible intervals for the same target")
        return self


class Artifact(ContractModel):
    uri: Text
    media_type: Text
    description: str | None = None
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None


class Tournament(ContractModel):
    id: Identifier
    season: Annotated[int, Field(ge=1900, le=2200)]
    competition: Text
    format: Literal["single_elimination"] = "single_elimination"
    topology_status: Literal["available", "unavailable"]
    bracket_team_count: PositiveInt | None = None
    includes_first_four: bool = False


class Team(ContractModel):
    id: Identifier
    name: Text


class Participant(ContractModel):
    team_id: Identifier | None = None
    source_game_id: Identifier | None = None

    @model_validator(mode="after")
    def one_source(self) -> Self:
        if (self.team_id is None) == (self.source_game_id is None):
            raise ValueError("A participant must reference exactly one team or source game")
        return self


class Game(ContractModel):
    id: Identifier
    slot: PositiveInt
    round_number: PositiveInt
    round_points: Nonnegative
    participants: Annotated[list[Participant], Field(min_length=2, max_length=2)]
    scheduled_start_utc: AwareDatetime | None = None
    actual_winner_team_id: Identifier | None = None


class PredictionRun(ContractModel):
    id: Identifier
    tournament_id: Identifier
    method: Text = Field(description="Current producers: ridge_residual_bootstrap and shap_ensemble. Future producers may use other names.")
    model_name: Text
    model_version: str | None = None
    generated_at: AwareDatetime
    n_simulations: PositiveInt | None = None
    seed: int | None = None
    source_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)


class ThresholdProbability(ContractModel):
    minimum_wins: PositiveInt
    probability: Probability | None


class TeamProbabilities(ContractModel):
    success_score_threshold: list[ThresholdProbability] = Field(default_factory=list)
    calibrated_success_score_threshold: list[ThresholdProbability] = Field(default_factory=list)
    bracket_tree: list[ThresholdProbability] = Field(default_factory=list)

    @model_validator(mode="after")
    def monotone(self) -> Self:
        for name in type(self).model_fields:
            values = getattr(self, name)
            thresholds = [v.minimum_wins for v in values]
            if thresholds != sorted(set(thresholds)):
                raise ValueError(f"{name} thresholds must be unique and increasing")
            known = [v.probability for v in values if v.probability is not None]
            if any(b > a + 1e-12 for a, b in zip(known, known[1:])):
                raise ValueError(f"{name} probabilities must be nonincreasing")
        return self


class TeamPrediction(ContractModel):
    run_id: Identifier
    team_id: Identifier
    statistics: dict[str, FiniteFloat | None] = Field(default_factory=dict)
    probabilities: TeamProbabilities = Field(default_factory=TeamProbabilities)
    intervals: list[Interval] = Field(default_factory=list)
    posterior: PosteriorSummary | None = None


class PickConfidence(ContractModel):
    value: Probability
    meaning: Literal["probability_team_wins_slot"] = "probability_team_wins_slot"
    method: Text
    intervals: list[Interval] = Field(default_factory=list)


class Pick(ContractModel):
    game_id: Identifier
    winner_team_id: Identifier
    confidence: PickConfidence
    public_pick_probability: Probability | None = None
    expected_points: Nonnegative | None = None
    leverage_ratio: Nonnegative | None = None
    actual_correct: bool | None = None


class BracketSummary(ContractModel):
    champion_team_id: Identifier
    expected_points: Nonnegative | None = None
    pool_win_probability: Probability | None = None
    top_10_percent_probability: Probability | None = None
    actual_points: Nonnegative | None = None
    pool_size: PositiveInt | None = None
    extensions: dict[str, JsonValue] = Field(default_factory=dict)


class Bracket(ContractModel):
    id: Identifier
    run_id: Identifier
    tournament_id: Identifier
    strategy: Text
    picks: Annotated[list[Pick], Field(min_length=1)]
    summary: BracketSummary


class MarketOutcome(ContractModel):
    team_name: Text
    team_id: Identifier | None = None
    team_mapping: Literal["exact_name", "alias", "unmatched"]
    market_ticker: Text
    win_probability: Probability | None = None
    midpoint_probability: Probability | None = None
    bid_probability: Probability | None = None
    ask_probability: Probability | None = None
    last_trade_probability: Probability | None = None
    probability_basis: Text
    quote_asof_utc: AwareDatetime | None = None
    seconds_before_start: Nonnegative | None = None
    data_status: Text

    @model_validator(mode="after")
    def mapping_consistent(self) -> Self:
        if (self.team_id is None) != (self.team_mapping == "unmatched"):
            raise ValueError("Unmatched outcomes must have null team_id; matched outcomes need a team_id")
        return self


class MarketSnapshot(ContractModel):
    id: Identifier
    event_ticker: Text
    provider_game_id: Text
    game_id: Identifier | None = None
    title: str | None = None
    competition: Text
    scheduled_start_utc: AwareDatetime
    requested_snapshot_utc: AwareDatetime | None = None
    game_status: Text
    markets_for_same_game: PositiveInt | None = None
    timing_basis: str | None = None
    timing_source_url: str | None = None
    price_basis: str | None = None
    source_url: str | None = None
    exception_source_url: str | None = None
    outcomes: Annotated[list[MarketOutcome], Field(min_length=2, max_length=2)]
    source_fields: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def pregame_timing(self) -> Self:
        if self.requested_snapshot_utc and self.requested_snapshot_utc > self.scheduled_start_utc:
            raise ValueError("Requested pregame snapshot is after scheduled start")
        if self.outcomes[0].market_ticker == self.outcomes[1].market_ticker:
            raise ValueError("Outcomes must use different market tickers")
        for outcome in self.outcomes:
            if outcome.quote_asof_utc:
                if outcome.quote_asof_utc > (self.requested_snapshot_utc or self.scheduled_start_utc):
                    raise ValueError("Quote is after the requested pregame cutoff")
                seconds = (self.scheduled_start_utc - outcome.quote_asof_utc).total_seconds()
                if outcome.seconds_before_start is not None and abs(seconds - outcome.seconds_before_start) > 1:
                    raise ValueError("seconds_before_start disagrees with quote timestamp")
        return self


class ExcludedMarketRow(ContractModel):
    row_number: PositiveInt
    event_ticker: str | None = None
    reason: Literal["other_competition", "missing_start_time", "other_season"]


class MarketImportSummary(ContractModel):
    rows_read: Annotated[int, Field(ge=0)]
    included: Annotated[int, Field(ge=0)]
    excluded: list[ExcludedMarketRow] = Field(default_factory=list)
    unmatched_outcomes: Annotated[int, Field(ge=0)]


class MarketDataset(ContractModel):
    provider: Literal["kalshi"] = "kalshi"
    market_type: Literal["game_winner"] = "game_winner"
    source: Artifact
    imported_at: AwareDatetime
    season_start_utc: AwareDatetime
    season_end_exclusive_utc: AwareDatetime
    import_summary: MarketImportSummary
    snapshots: list[MarketSnapshot]

    @model_validator(mode="after")
    def import_counts(self) -> Self:
        summary = self.import_summary
        if summary.included != len(self.snapshots) or summary.rows_read != summary.included + len(summary.excluded):
            raise ValueError("Market import counts do not match records")
        unmatched = sum(o.team_id is None for s in self.snapshots for o in s.outcomes)
        if unmatched != summary.unmatched_outcomes:
            raise ValueError("Unmatched outcome count is incorrect")
        for snapshot in self.snapshots:
            if not self.season_start_utc <= snapshot.scheduled_start_utc < self.season_end_exclusive_utc:
                raise ValueError("Market snapshot is outside season window")
        return self


def index_unique(items, label):
    ids = [item.id for item in items]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {duplicates}")
    return {item.id: item for item in items}


class ForecastDocument(ContractModel):
    """One tournament, any number of prediction runs/strategies, optional market data."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    tournament: Tournament
    teams: list[Team]
    games: list[Game] = Field(default_factory=list)
    prediction_runs: list[PredictionRun] = Field(default_factory=list)
    team_predictions: list[TeamPrediction] = Field(default_factory=list)
    brackets: list[Bracket] = Field(default_factory=list)
    market_data: MarketDataset | None = None

    @model_validator(mode="after")
    def references_and_brackets(self) -> Self:
        teams = index_unique(self.teams, "team")
        games = index_unique(self.games, "game")
        runs = index_unique(self.prediction_runs, "run")
        index_unique(self.brackets, "bracket")
        if any(r.tournament_id != self.tournament.id for r in runs.values()):
            raise ValueError("Run references a different tournament")
        if (self.tournament.topology_status == "available") != bool(games):
            raise ValueError("Topology status disagrees with available games")
        pairs = [(p.run_id, p.team_id) for p in self.team_predictions]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate team prediction for a run")
        for prediction in self.team_predictions:
            if prediction.run_id not in runs or prediction.team_id not in teams:
                raise ValueError("Team prediction references unknown run or team")
        eligible, consumed, opening_teams = {}, Counter(), []
        slots = [g.slot for g in games.values()]
        if len(slots) != len(set(slots)):
            raise ValueError("Duplicate game slots")
        for game in sorted(games.values(), key=lambda g: g.slot):
            branches = []
            for participant in game.participants:
                if participant.team_id is not None:
                    if participant.team_id not in teams:
                        raise ValueError("Game references unknown team")
                    opening_teams.append(participant.team_id)
                    branches.append({participant.team_id})
                else:
                    source = participant.source_game_id
                    if source not in eligible or games[source].round_number >= game.round_number:
                        raise ValueError("Game references a future, cyclic, or invalid source game")
                    consumed[source] += 1
                    branches.append(eligible[source])
            if branches[0] & branches[1]:
                raise ValueError("Game participant branches overlap")
            eligible[game.id] = branches[0] | branches[1]
            if game.actual_winner_team_id and game.actual_winner_team_id not in eligible[game.id]:
                raise ValueError("Actual winner is not eligible for game")
        if games:
            finals = set(games) - set(consumed)
            if len(finals) != 1 or any(n != 1 for n in consumed.values()):
                raise ValueError("Topology must be a connected single-elimination tree")
            if len(opening_teams) != len(set(opening_teams)) or len(games) != len(opening_teams) - 1:
                raise ValueError("Teams must enter the bracket exactly once")
            if self.tournament.bracket_team_count != len(opening_teams):
                raise ValueError("Bracket team count disagrees with topology")
        for bracket in self.brackets:
            if bracket.run_id not in runs or bracket.tournament_id != self.tournament.id:
                raise ValueError("Bracket references unknown run or tournament")
            picks = {p.game_id: p for p in bracket.picks}
            if len(picks) != len(bracket.picks) or set(picks) != set(games):
                raise ValueError("Bracket must pick each game exactly once")
            for game_id, pick in picks.items():
                allowed = {p.team_id or picks[p.source_game_id].winner_team_id for p in games[game_id].participants}
                if pick.winner_team_id not in allowed:
                    raise ValueError(f"Inconsistent pick for {game_id}")
            final_id = next(iter(set(games) - set(consumed)))
            if bracket.summary.champion_team_id != picks[final_id].winner_team_id:
                raise ValueError("Bracket champion disagrees with final pick")
        if self.market_data:
            expected_start = datetime(self.tournament.season - 1, 7, 1, tzinfo=timezone.utc)
            expected_end = datetime(self.tournament.season, 7, 1, tzinfo=timezone.utc)
            if (self.market_data.season_start_utc != expected_start
                    or self.market_data.season_end_exclusive_utc != expected_end):
                raise ValueError("Market season window differs from tournament season")
            index_unique(self.market_data.snapshots, "market snapshot")
            for snapshot in self.market_data.snapshots:
                if snapshot.competition != self.tournament.competition:
                    raise ValueError("Market competition differs from tournament")
                mapped = {o.team_id for o in snapshot.outcomes if o.team_id is not None}
                known_ids = [o.team_id for o in snapshot.outcomes if o.team_id is not None]
                if len(known_ids) != len(set(known_ids)):
                    raise ValueError("Market outcomes map to the same team")
                if any(t not in teams for t in mapped):
                    raise ValueError("Market references unknown team")
                if snapshot.game_id is not None:
                    if snapshot.game_id not in games or len(mapped) != 2 or not mapped <= eligible[snapshot.game_id]:
                        raise ValueError("Market game mapping has unknown or incompatible teams")
                    for participant in games[snapshot.game_id].participants:
                        branch = {participant.team_id} if participant.team_id else eligible[participant.source_game_id]
                        if len(mapped & branch) != 1:
                            raise ValueError("Market teams cannot meet in the mapped game")
        return self
