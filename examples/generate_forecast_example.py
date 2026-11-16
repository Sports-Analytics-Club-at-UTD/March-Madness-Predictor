"""Generate a small, entirely synthetic example of the public contract.

Run from the repository root: python -m examples.generate_forecast_example
"""

from pathlib import Path

from backend_data.models import ForecastDocument


def example_document():
    tournament = "example-men-2026"
    first_run, second_run = "example-shap-run", "example-bayesian-run"
    teams = [{"id": f"example-team-{i}", "name": f"Example Team {i}"} for i in range(4)]
    games = [
        {"id": f"{tournament}-g1", "slot": 1, "round_number": 1, "round_points": 10,
         "participants": [{"team_id": teams[i]["id"]} for i in (0, 1)]},
        {"id": f"{tournament}-g2", "slot": 2, "round_number": 1, "round_points": 10,
         "participants": [{"team_id": teams[i]["id"]} for i in (2, 3)]},
        {"id": f"{tournament}-g3", "slot": 3, "round_number": 2, "round_points": 20,
         "participants": [{"source_game_id": f"{tournament}-g{i}"} for i in (1, 2)]},
    ]
    brackets = []
    for strategy, chosen in (("max_ev", (0, 2, 0)), ("leverage", (1, 3, 3))):
        brackets.append({
            "id": f"{first_run}-{strategy}", "run_id": first_run, "tournament_id": tournament,
            "strategy": strategy,
            "picks": [{"game_id": game["id"], "winner_team_id": teams[team]["id"],
                       "confidence": {"value": .4, "method": "illustrative_slot_simulation"},
                       "expected_points": game["round_points"] * .4}
                      for game, team in zip(games, chosen)],
            "summary": {"champion_team_id": teams[chosen[-1]]["id"], "expected_points": 16.},
        })
    return ForecastDocument.model_validate({
        "tournament": {"id": tournament, "season": 2026, "competition": "Men's college basketball",
                       "topology_status": "available", "bracket_team_count": 4},
        "teams": teams, "games": games, "brackets": brackets,
        "prediction_runs": [{
            "id": run, "tournament_id": tournament, "method": method, "model_name": "synthetic_example",
            "generated_at": "2026-03-18T12:00:00Z", "source_metadata": {"example_only": True},
        } for run, method in ((first_run, "shap_example"), (second_run, "bayesian_example"))],
        "team_predictions": [
            {"run_id": first_run, "team_id": teams[0]["id"], "statistics": {"simulated_mean": 1.1},
             "probabilities": {"bracket_tree": [{"minimum_wins": 1, "probability": .6},
                                                 {"minimum_wins": 2, "probability": .4}]}},
            {"run_id": second_run, "team_id": teams[0]["id"],
             "posterior": {"target": "tournament_success_score", "mean": 1.2, "std": .3,
                           "intervals": [{"kind": "credible", "target": "tournament_success_score",
                                          "level": .9, "lower": .7, "upper": 1.7,
                                          "method": "illustrative_posterior_quantiles"}]}},
        ],
        "market_data": {
            "source": {"uri": "synthetic-example.csv", "media_type": "text/csv"},
            "imported_at": "2026-03-20T12:00:00Z",
            "season_start_utc": "2025-07-01T00:00:00Z", "season_end_exclusive_utc": "2026-07-01T00:00:00Z",
            "import_summary": {"rows_read": 1, "included": 1, "unmatched_outcomes": 0},
            "snapshots": [{
                "id": "example-snapshot", "event_ticker": "EXAMPLE-EVENT", "provider_game_id": "EXAMPLE-PROVIDER-GAME",
                "game_id": games[0]["id"], "title": "Synthetic game quote", "competition": "Men's college basketball",
                "scheduled_start_utc": "2026-03-19T16:00:00Z", "requested_snapshot_utc": "2026-03-19T15:59:00Z",
                "game_status": "example_only", "outcomes": [{
                    "team_name": teams[i]["name"], "team_id": teams[i]["id"], "team_mapping": "exact_name",
                    "market_ticker": f"EXAMPLE-MARKET-{i}", "win_probability": p,
                    "probability_basis": "bid_ask_midpoint", "data_status": status,
                    "quote_asof_utc": timestamp, "seconds_before_start": age,
                } for i, p, status, timestamp, age in (
                    (0, .5, "ok", "2026-03-19T15:59:00Z", 60),
                    (1, .51, "older_than_requested_minute", "2026-03-19T15:55:00Z", 300),
                )],
            }],
        },
    })


if __name__ == "__main__":
    path = Path(__file__).with_name("forecast.example.json")
    path.write_text(example_document().model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(path)
