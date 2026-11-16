import csv
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from bayesian_evaluation import SeasonSpec, load_config, load_matchup_scores
from bayesian_ratings import utc_timestamp
from historical_matchup_scores import (
    build_cutoff_features, export_historical_scores, select_earlier_weights,
)


class HistoricalScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.manifest = self.root / "manifests" / "2010" / "manifest.json"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(json.dumps(dict(
            season=2010, status="complete", statistics_scope="regular season, before NCAA tournament",
            statistics_cutoff="2010-03-17T00:00:00Z")))
        self.spec = SeasonSpec(2010, utc_timestamp("2010-03-18T00:00:00Z"),
                               utc_timestamp("2010-05-01T00:00:00Z"), 2)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "seasons": [dict(season=2010, cutoff=self.spec.cutoff.isoformat(),
                             until=self.spec.until.isoformat(), expected_games=2)],
            "evaluation_seasons": [2010], "matchup_members": ["ridge"],
        }))
        for team, points in ((10, 90), (20, 70), (30, 60)):
            folder = self.data / "2010" / f"team{team}"
            self.write_csv(folder / f"team{team}_game_stats.csv", [dict(
                gameId=i, season=2010, teamId=team, team=f"Team {team}", seasonType="regular",
                startDate=f"2010-01-0{i + 1}T00:00:00Z", status="final",
                **{"teamStats.points.total": points + i, "homeTeamEloEnd": 9999 + i},
            ) for i in range(2)])
            self.write_csv(folder / f"team{team}_players_season_stats.csv", [dict(
                season=2010, teamId=team, name=f"P{i}", minutes=100 + team + i,
                points=points + i, games=20, athleteId=team * 100 + i,
            ) for i in range(6)])
        self.weights_path = self.write_fold(2009, [2005, 2006, 2007, 2008])

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_fold(self, validation, training, *, features=None):
        names = features or ["game_mean_teamStats.points.total", "starters_mean_points"]
        folder = self.models / "ridge" / f"Train_{min(training)}_to_{max(training)}_Val_{validation}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "shap_feature_importance.csv"
        self.write_csv(path, [dict(feature=name, shap_importance=i + 1) for i, name in enumerate(names)])
        joblib.dump(dict(training_seasons=training, validation_season=validation, feature_names=names),
                    folder / "ridge_bundle.joblib")
        return path

    def export(self, name="export"):
        return export_historical_scores(self.config, self.root / name, members=["ridge"],
                                         data_dir=self.data, training_output_dir=self.models,
                                         manifest_dir=self.root / "manifests")

    def test_export_is_accepted_by_evaluator_with_traceable_earlier_weights(self):
        self.write_fold(2010, [2005, 2006, 2007, 2008, 2009])
        self.write_fold(2011, [2005, 2006, 2007, 2008, 2009, 2010])
        path = self.export()
        scores = load_matchup_scores(path, load_config(self.config)[0])
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[2010, "cbbd:10"], scores[2010, "cbbd:20"])
        provenance = json.loads(path.with_name("provenance.json").read_text())
        artifact = provenance["seasons"]["2010"]["members"][0]
        self.assertEqual(artifact["validation_season"], 2009)
        self.assertEqual(artifact["producer_through_season"], 2009)
        self.assertEqual(provenance["members"], ["ridge"])
        self.assertEqual(artifact["missing_weight_fraction"], 0)
        with self.assertRaises(FileExistsError):
            self.export()

    def test_future_text_and_nonregular_rows_cannot_change_historical_features(self):
        before, _, _ = build_cutoff_features(self.data, self.spec, self.manifest)
        path = self.data / "2010" / "team10" / "team10_game_stats.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows += [dict(rows[0], startDate="2010-04-01T00:00:00Z", teamId="", **{"teamStats.points.total": "future poison"}),
                 dict(rows[0], seasonType="postseason", **{"teamStats.points.total": "postseason poison"})]
        self.write_csv(path, rows)
        after, _, audit = build_cutoff_features(self.data, self.spec, self.manifest)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(audit["excluded_rows"]["cutoff"], 1)
        self.assertEqual(audit["excluded_rows"]["nonregular"], 1)
        self.assertFalse(any("EloEnd" in name for name in after))

    def test_fallback_to_same_season_or_future_fold_is_rejected(self):
        self.weights_path.unlink()
        self.write_fold(2010, [2005, 2006, 2007, 2008, 2009])
        self.write_fold(2011, [2005, 2006, 2007, 2008, 2009, 2010])
        with self.assertRaisesRegex(ValueError, "no strictly earlier"):
            select_earlier_weights(self.models, "ridge", 2010)

    def test_inconsistent_artifact_training_metadata_is_rejected(self):
        bundle_path = self.weights_path.with_name("ridge_bundle.joblib")
        bundle = joblib.load(bundle_path)
        bundle["training_seasons"].append(2010)
        joblib.dump(bundle, bundle_path)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.export()

    def test_manifest_scope_and_availability_are_required_for_player_aggregates(self):
        original = json.loads(self.manifest.read_text())
        for change in ({"statistics_cutoff": self.spec.cutoff.isoformat()},
                       {"status": "incomplete"}, {"statistics_scope": "season end"}):
            self.manifest.write_text(json.dumps(dict(original, **change)))
            with self.assertRaises(ValueError):
                build_cutoff_features(self.data, self.spec, self.manifest)

    def test_no_usable_weights_and_changed_fixed_membership_fail(self):
        self.write_fold(2009, [2005, 2006, 2007, 2008], features=["missing_feature"])
        with self.assertRaisesRegex(ValueError, "no usable"):
            self.export()
        with self.assertRaisesRegex(ValueError, "fixed membership"):
            export_historical_scores(self.config, self.root / "other", members=["ridge", "lasso"])

    def test_missing_player_cache_fails_instead_of_silently_dropping_team(self):
        path = self.data / "2010" / "team10" / "team10_players_season_stats.csv"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing player statistics"):
            self.export()


if __name__ == "__main__":
    unittest.main()
