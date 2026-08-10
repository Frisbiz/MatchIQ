import importlib.util
import io
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd


def load_app_module():
    module_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("matchiq_app_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    # Importing app.py starts the production preload thread. Replace Thread so
    # tests can exercise the module without network side effects at import time.
    with patch("threading.Thread"):
        spec.loader.exec_module(module)

    return module


class SeasonUpdateTests(TestCase):
    def setUp(self):
        self.app = load_app_module()

    def test_2026_27_team_lists_replace_relegated_clubs(self):
        expected = {
            "Premier League": {
                "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
                "Chelsea", "Coventry", "Crystal Palace", "Everton", "Fulham",
                "Hull", "Ipswich", "Leeds", "Liverpool", "Man City",
                "Man United", "Newcastle", "Nottingham Forest", "Sunderland",
                "Tottenham",
            },
            "La Liga": {
                "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona",
                "Betis", "Celta Vigo", "Deportivo La Coruna", "Elche",
                "Espanyol", "Getafe", "Levante", "Malaga", "Osasuna",
                "Racing Santander", "Rayo Vallecano", "Real Madrid",
                "Real Sociedad", "Sevilla", "Valencia", "Villarreal",
            },
            "Serie A": {
                "Atalanta", "Bologna", "Cagliari", "Como", "Fiorentina",
                "Frosinone", "Genoa", "Inter Milan", "Juventus", "Lazio",
                "Lecce", "Milan", "Monza", "Napoli", "Parma", "Roma",
                "Sassuolo", "Torino", "Udinese", "Venezia",
            },
            "Bundesliga": {
                "Augsburg", "Union Berlin", "Werder Bremen", "Dortmund",
                "Eintracht Frankfurt", "Freiburg", "Hamburg", "Hoffenheim",
                "Koln", "RB Leipzig", "Leverkusen", "Mainz",
                "Monchengladbach", "Bayern Munich", "Schalke", "Stuttgart",
                "Elversberg", "Paderborn",
            },
            "Ligue 1": {
                "Angers", "Auxerre", "Brest", "Le Havre", "Le Mans",
                "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Monaco",
                "Nice", "Paris FC", "Paris SG", "Rennes", "Strasbourg",
                "Toulouse", "Troyes",
            },
        }

        for league, expected_teams in expected.items():
            with self.subTest(league=league):
                self.assertEqual(set(self.app.LEAGUE_DATA[league]["teams"]), expected_teams)

    def test_football_data_names_normalize_to_display_names(self):
        raw = pd.DataFrame(
            {
                "HomeTeam": ["Ath Bilbao", "Inter", "Ein Frankfurt", "M'gladbach"],
                "AwayTeam": ["Sociedad", "Milan", "FC Koln", "Paris St-G"],
                "FTHG": [1, 2, 3, 1],
                "FTAG": [1, 0, 2, 2],
                "FTR": ["D", "H", "H", "A"],
            }
        )

        normalized = self.app.normalize_team_names(raw)

        self.assertEqual(
            list(normalized["HomeTeam"]),
            ["Athletic Bilbao", "Inter Milan", "Eintracht Frankfurt", "Monchengladbach"],
        )
        self.assertEqual(
            list(normalized["AwayTeam"]),
            ["Real Sociedad", "Milan", "Koln", "Paris SG"],
        )

    def test_fetch_data_skips_wrong_league_csv_that_has_a_valid_url(self):
        wrong_league = pd.DataFrame(
            {
                "HomeTeam": ["Arbroath", "Dunfermline"],
                "AwayTeam": ["Ayr", "Inverness C"],
                "FTHG": [1, 0],
                "FTAG": [0, 2],
                "FTR": ["H", "A"],
                "Date": ["08/08/26", "09/08/26"],
            }
        )

        with patch.object(self.app, "SEASONS", [("2627", "2026-27")]):
            with patch.object(self.app.os.path, "exists", return_value=False):
                with patch.object(self.app, "read_csv_with_timeout", return_value=wrong_league):
                    self.assertIsNone(self.app.fetch_data("La Liga"))

    def test_fetch_data_logs_are_safe_for_windows_stdout(self):
        valid_rows = pd.DataFrame(
            {
                "HomeTeam": ["Arsenal", "Liverpool", "Everton"],
                "AwayTeam": ["Chelsea", "Man City", "Fulham"],
                "FTHG": [2, 1, 0],
                "FTAG": [1, 1, 0],
                "FTR": ["H", "D", "D"],
                "Date": ["01/08/25", "02/08/25", "03/08/25"],
            }
        )
        cp1252_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

        with patch.object(self.app, "SEASONS", [("2526", "2025-26")]):
            with patch.object(self.app.os.path, "exists", return_value=False):
                with patch.object(self.app, "read_csv_with_timeout", return_value=valid_rows):
                    with patch("sys.stdout", cp1252_stdout):
                        data = self.app.fetch_data("Premier League")

        self.assertEqual(len(data), 3)

    def test_previous_season_rows_are_not_treated_as_current_season(self):
        previous_season = pd.DataFrame(
            {
                "HomeTeam": ["Arsenal", "Liverpool", "Everton"],
                "AwayTeam": ["Chelsea", "Man City", "Fulham"],
                "FTHG": [2, 1, 0],
                "FTAG": [1, 1, 0],
                "FTR": ["H", "D", "D"],
                "Date": ["01/08/25", "02/08/25", "03/08/25"],
            }
        )

        with patch.object(self.app, "SEASONS", [("2526", "2025-26")]):
            with patch.object(self.app.os.path, "exists", return_value=False):
                with patch.object(self.app, "read_csv_with_timeout", return_value=previous_season):
                    data = self.app.fetch_data("Premier League")

        current_rows = data[data["SeasonKey"] == self.app.CURRENT_SEASON_KEY]
        self.assertEqual(len(current_rows), 0)

    def test_current_season_filter_does_not_fallback_to_latest_snapshot(self):
        previous_season = pd.DataFrame(
            {
                "Season": ["2025-26", "2025-26", "2025-26"],
                "SeasonKey": ["26", "26", "26"],
                "HomeTeam": ["Arsenal", "Liverpool", "Everton"],
                "AwayTeam": ["Chelsea", "Man City", "Fulham"],
                "FTHG": [2, 1, 0],
                "FTAG": [1, 1, 0],
                "FTR": ["H", "D", "D"],
                "Date": ["01/08/25", "02/08/25", "03/08/25"],
            }
        )

        current = self.app._current_season_df(previous_season)

        self.assertEqual(len(current), 0)

    def test_projected_standings_start_from_zero_when_current_season_has_no_results(self):
        class DrawModel:
            def predict(self, home, away, strength_profile=None):
                return {
                    "home_goals": 1,
                    "away_goals": 1,
                    "home_prob": 0.0,
                    "draw_prob": 1.0,
                    "away_prob": 0.0,
                }

        teams = ["Arsenal", "Chelsea", "Liverpool"]
        standings = self.app.simulate_remaining_season_standings(
            DrawModel(),
            pd.DataFrame(columns=["Season", "SeasonKey", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]),
            teams,
            league="Premier League",
            n_sim=1,
        )

        self.assertEqual({row["team"] for row in standings}, set(teams))
        self.assertTrue(all(row["current_points"] == 0 for row in standings))
        self.assertTrue(all(row["played"] == 0 for row in standings))

    def test_prediction_profile_falls_back_for_promoted_teams_without_recent_profile(self):
        model = self.app.EnhancedPoissonModel()
        model.global_avg = 1.3
        model.home_advantage = 0.25
        model.rho = 0.03
        model.team_attack = {"Arsenal": 1.2, "Coventry": 1.0}
        model.team_defense = {"Arsenal": 0.8, "Coventry": 1.1}

        result = model.predict(
            "Arsenal",
            "Coventry",
            strength_profile={
                "attack": {"Arsenal": 1.15},
                "defense": {"Arsenal": 0.85},
                "global_avg": 1.25,
            },
        )

        self.assertIsNotNone(result)
        self.assertIn("home_prob", result)
