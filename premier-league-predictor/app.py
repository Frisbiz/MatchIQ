from flask import Flask, render_template, jsonify, request, make_response
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json
import csv
import threading
import warnings
from urllib.request import urlopen
from urllib.parse import urlencode
from io import BytesIO
warnings.filterwarnings('ignore')

app = Flask(__name__)
APP_VERSION = 'bg-refresh-v16'

# Manual CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== ENHANCED POISSON MODEL ====================

class EnhancedPoissonModel:
    """Enhanced Poisson with weighted seasons, form, and home advantage"""
    
    def __init__(self):
        self.team_attack = {}
        self.team_defense = {}
        self.home_advantage = 0.0
        self.rho = 0.0
        self.n_teams = 0
        self.global_avg = 0.0
        self.teams_list = []
        
    def fit(self, df, teams):
        """Fit model using simple weighted averages - fast version"""
        self.n_teams = len(teams)
        self.teams_list = teams
        
        # Calculate global average (fast vectorized)
        self.global_avg = (df['FTHG'].sum() + df['FTAG'].sum()) / (len(df) * 2)
        
        # Simple attack/defense based on weighted averages
        attack = {}
        defense = {}
        
        for team in teams:
            home_df = df[df['HomeTeam'] == team]
            away_df = df[df['AwayTeam'] == team]
            
            if len(home_df) > 0:
                home_gs = np.average(home_df['FTHG'].values, weights=home_df['Weight'].values)
                home_gc = np.average(home_df['FTAG'].values, weights=home_df['Weight'].values)
            else:
                home_gs = home_gc = self.global_avg
                
            if len(away_df) > 0:
                away_gs = np.average(away_df['FTAG'].values, weights=away_df['Weight'].values)
                away_gc = np.average(away_df['FTHG'].values, weights=away_df['Weight'].values)
            else:
                away_gs = away_gc = self.global_avg
            
            # Attack = goals scored / global avg
            # Defense = goals conceded / global avg
            attack[team] = (home_gs + away_gs) / (2 * self.global_avg)
            defense[team] = (home_gc + away_gc) / (2 * self.global_avg)
        
        self.team_attack = attack
        self.team_defense = defense
        
        # Estimate home advantage from data
        home_wins = (df['FTHG'] > df['FTAG']).sum()
        draws = (df['FTHG'] == df['FTAG']).sum()
        self.home_advantage = 0.35  # Standard home advantage
        
        # Estimate rho (correlation for low scores)
        self.rho = 0.03  # Small positive correlation
        
        print(f"✓ Enhanced Poisson model fitted for {len(teams)} teams")
        print(f"  Global avg: {self.global_avg:.3f}, Home adv: {self.home_advantage:.3f}")
    
    def predict(self, home_team, away_team, exclude_draw=False):
        """Predict match using enhanced Poisson"""
        if home_team not in self.team_attack or away_team not in self.team_attack:
            return None
        
        # Expected goals with team strengths
        lam = float(self.global_avg * self.team_attack[home_team] / self.team_defense[away_team] * np.exp(self.home_advantage))
        mu = float(self.global_avg * self.team_attack[away_team] / self.team_defense[home_team])
        
        # Bound
        lam = max(0.3, min(lam, 4.0))
        mu = max(0.3, min(mu, 4.0))
        
        # Calculate score probabilities
        score_probs = {}
        home_win_total = 0
        away_win_total = 0
        draw_total = 0
        
        for h in range(7):
            for a in range(7):
                prob = poisson.pmf(h, lam) * poisson.pmf(a, mu)
                
                # Adjustment for low scores (simplified DC)
                if (h == 0 and a == 0) or (h == 1 and a == 0) or (h == 0 and a == 1):
                    prob *= (1 + self.rho)
                
                prob = max(0, prob)
                score_probs[(h, a)] = prob
                
                if h > a:
                    home_win_total += prob
                elif a > h:
                    away_win_total += prob
                else:
                    draw_total += prob
        
        # Normalize
        total = home_win_total + away_win_total + draw_total
        if total > 0:
            home_win_total /= total
            away_win_total /= total
            draw_total /= total
        
        # Find most likely score
        best_score = (1, 1)
        best_prob = 0
        for (h, a), prob in score_probs.items():
            if exclude_draw and h == a:
                continue
            if prob > best_prob:
                best_prob = prob
                best_score = (h, a)
        
        # Confidence
        confidence = min(0.95, best_prob * 3 + 0.3)
        
        return {
            'home_goals': best_score[0],
            'away_goals': best_score[1],
            'home_prob': home_win_total,
            'draw_prob': draw_total,
            'away_prob': away_win_total,
            'confidence': confidence,
            'expected_goals': {'home': lam, 'away': mu}
        }


# Team colors
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BWE5", "Bournemouth": "#B50127",
    "Brentford": "#E30613", "Brighton": "#0057B8", "Chelsea": "#034694",
    "Crystal Palace": "#1B458F", "Everton": "#003399", "Fulham": "#CC0000",
    "Ipswich": "#00A650", "Leicester": "#00308F", "Liverpool": "#C8102E",
    "Man City": "#6CABDD", "Man United": "#DA291C", "Newcastle": "#241F20",
    "Nottingham Forest": "#DD0000", "Southampton": "#D70027", "Tottenham": "#132257",
    "West Ham": "#7A263A", "Wolves": "#FDB912",
    "Barcelona": "#A50044", "Real Madrid": "#FFFFFF", "Atletico Madrid": "#CB3524",
    "Bayern Munich": "#DC052D", "Dortmund": "#FDE100", "PSG": "#004170",
    "Juventus": "#000000", "Milan": "#FB090B", "Inter Milan": "#010E80",
    "Roma": "#9d0000", "Napoli": "#0073CF", "Lyon": "#DA291C", "Marseille": "#0099CB"
}

# Current teams by league
premier_league_teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Sunderland", "Tottenham", "West Ham", "Wolves"
]

la_liga_teams = [
    "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Celta Vigo",
    "Elche", "Espanyol", "Getafe", "Girona", "Levante",
    "Mallorca", "Osasuna", "Oviedo", "Rayo Vallecano", "Betis",
    "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
]

serie_a_teams = [
    "Atalanta", "Bologna", "Cagliari", "Como", "Cremonese",
    "Fiorentina", "Genoa", "Verona", "Inter Milan", "Juventus",
    "Lazio", "Lecce", "Milan", "Napoli", "Parma",
    "Pisa", "Roma", "Sassuolo", "Torino", "Udinese"
]

bundesliga_teams = [
    "Augsburg", "Union Berlin", "Werder Bremen", "Dortmund", "Eintracht Frankfurt",
    "Freiburg", "Hamburg", "Heidenheim", "Hoffenheim", "Koln",
    "RB Leipzig", "Leverkusen", "Mainz", "Monchengladbach", "Bayern Munich",
    "St Pauli", "Stuttgart", "Wolfsburg"
]

ligue_1_teams = [
    "Angers", "Auxerre", "Brest", "Le Havre", "Lens", "Lille",
    "Lorient", "Lyon", "Marseille", "Metz", "Monaco", "Nantes",
    "Nice", "Paris FC", "Paris SG", "Rennes", "Strasbourg", "Toulouse"
]

# Leagues
LEAGUE_DATA = {
    "Premier League": {"country": "England", "code": "E0", "teams": premier_league_teams, "yahoo_code": "soccer.l.fbgb"},
    "La Liga": {"country": "Spain", "code": "SP1", "teams": la_liga_teams, "yahoo_code": None},
    "Serie A": {"country": "Italy", "code": "I1", "teams": serie_a_teams, "yahoo_code": None},
    "Bundesliga": {"country": "Germany", "code": "D1", "teams": bundesliga_teams, "yahoo_code": None},
    "Ligue 1": {"country": "France", "code": "F1", "teams": ligue_1_teams, "yahoo_code": None},
}

# Season weights
SEASON_WEIGHTS = {
    "2425": 2.5, "2324": 2.0, "2223": 1.5, "2122": 1.2, "2021": 1.0,
    "1920": 0.9, "1819": 0.8, "1718": 0.7, "1617": 0.6, "1516": 0.5, "1415": 0.4
}

def fetch_yahoo_scoreboard(league, week):
    league_info = LEAGUE_DATA.get(league, LEAGUE_DATA["Premier League"])
    yahoo_code = league_info.get("yahoo_code")
    if not yahoo_code:
        return None

    params = {
        'lang': 'en-US',
        'ysp_redesign': '1',
        'ysp_platform': 'desktop',
        'leagues': yahoo_code,
        'week': week,
        'sched_states': '2',
        'v': '2',
        'ysp_enable_last_update': '1',
    }
    url = f"https://api-secure.sports.yahoo.com/v1/editorial/s/scoreboard?{urlencode(params)}"

    try:
        with urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"⚠️ Yahoo fetch failed for {league} week {week}: {e}")
        return None


def read_csv_with_timeout(url, timeout=10):
    with urlopen(url, timeout=timeout) as resp:
        return pd.read_csv(BytesIO(resp.read()))


def read_local_training_snapshot(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ['FTHG', 'FTAG', 'Weight']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fit_fast_model(df, teams):
    model = EnhancedPoissonModel()
    model.n_teams = len(teams)
    model.teams_list = teams
    attack_for = {team: 0.0 for team in teams}
    defense_against = {team: 0.0 for team in teams}
    weights = {team: 0.0 for team in teams}
    total_goals = 0.0
    total_rows = 0

    for row in df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Weight']].itertuples(index=False, name=None):
        home, away, home_goals, away_goals, weight = row
        try:
            home_goals = float(home_goals)
            away_goals = float(away_goals)
            weight = float(weight) if weight else 1.0
        except (TypeError, ValueError):
            continue
        total_goals += home_goals + away_goals
        total_rows += 1
        if home in weights:
            attack_for[home] += home_goals * weight
            defense_against[home] += away_goals * weight
            weights[home] += weight
        if away in weights:
            attack_for[away] += away_goals * weight
            defense_against[away] += home_goals * weight
            weights[away] += weight

    model.global_avg = total_goals / max(total_rows * 2, 1)
    model.home_advantage = 0.35
    model.rho = 0.03
    model.team_attack = {}
    model.team_defense = {}
    for team in teams:
        team_weight = max(weights.get(team, 0.0), 1.0)
        scored = attack_for.get(team, 0.0) / team_weight
        conceded = defense_against.get(team, 0.0) / team_weight
        model.team_attack[team] = max(0.3, scored / max(model.global_avg, 0.1))
        model.team_defense[team] = max(0.3, conceded / max(model.global_avg, 0.1))
    return model


def load_precomputed_model(league, teams):
    params_paths = [
        os.path.join(os.path.dirname(__file__), 'data', 'model-params.json'),
        os.path.join(os.getcwd(), 'data', 'model-params.json'),
        os.path.join(os.getcwd(), 'premier-league-predictor', 'data', 'model-params.json'),
    ]
    params = None
    for path in params_paths:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                params = json.load(f).get(league)
            break
    if not params:
        return None
    model = EnhancedPoissonModel()
    model.n_teams = len(teams)
    model.teams_list = teams
    model.global_avg = float(params.get('global_avg') or 1.3)
    model.home_advantage = 0.35
    model.rho = 0.03
    model.team_attack = {team: float(params.get('attack', {}).get(team, 1.0)) for team in teams}
    model.team_defense = {team: float(params.get('defense', {}).get(team, 1.0)) for team in teams}
    return {
        'model': model,
        'df': pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'SeasonKey']),
        'teams': list(teams),
        'team_stats': {team: params.get('team_stats', {}).get(team, {}) for team in teams},
        'standings': [row for row in params.get('standings', []) if row.get('team') in teams],
        'match_count': int(params.get('match_count') or 0),
        'data_latest_match_date': params.get('latest_match_date'),
    }


YAHOO_TEAM_NAME_MAP = {
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Brighton & Hove Albion': 'Brighton',
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Leeds United': 'Leeds',
    'Liverpool': 'Liverpool',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': 'Nottingham Forest',
    'Sunderland': 'Sunderland',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
}


def fetch_yahoo_current_results(league, max_weeks=38):
    import time
    league_info = LEAGUE_DATA.get(league, LEAGUE_DATA['Premier League'])
    if not league_info.get('yahoo_code'):
        return None

    rows = []
    deadline = time.time() + 30  # bail out after 30 seconds total
    for week in range(1, max_weeks + 1):
        if time.time() > deadline:
            print(f"⚠️ Yahoo fetch budget exceeded for {league}, stopping at week {week}")
            break
        payload = fetch_yahoo_scoreboard(league, week)
        if not payload:
            continue
        games = payload.get('service', {}).get('scoreboard', {}).get('games', {})
        if not isinstance(games, dict):
            continue

        for game in games.values():
            home_name = YAHOO_TEAM_NAME_MAP.get(game.get('home_team_name') or game.get('home_team_display_name') or '')
            away_name = YAHOO_TEAM_NAME_MAP.get(game.get('away_team_name') or game.get('away_team_display_name') or '')
            if not home_name or not away_name:
                continue

            status = game.get('status_type')
            home_goals = game.get('total_home_points')
            away_goals = game.get('total_away_points')
            if status != 'status.type.final' or home_goals is None or away_goals is None:
                continue

            try:
                hg = int(home_goals)
                ag = int(away_goals)
            except (ValueError, TypeError):
                continue

            rows.append({
                'HomeTeam': home_name,
                'AwayTeam': away_name,
                'FTHG': hg,
                'FTAG': ag,
                'FTR': 'H' if hg > ag else 'A' if ag > hg else 'D',
                'Week': int(game.get('week_number') or week),
            })

    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates(subset=['HomeTeam', 'AwayTeam'], keep='last')


def fetch_data(league="Premier League"):
    """Fetch league data"""
    league_info = LEAGUE_DATA.get(league, LEAGUE_DATA["Premier League"])
    code = league_info["code"]
    slug = re.sub(r'[^a-z0-9]+', '-', league.lower()).strip('-')
    snapshot_paths = [
        os.path.join(os.path.dirname(__file__), 'data', f'{slug}.csv'),
        os.path.join(os.getcwd(), 'data', f'{slug}.csv'),
        os.path.join(os.getcwd(), 'premier-league-predictor', 'data', f'{slug}.csv'),
    ]
    for snapshot_path in snapshot_paths:
        if os.path.exists(snapshot_path):
            print(f"✓ {league} local training snapshot: {snapshot_path}")
            try:
                mark_refresh_state(league, refresh_stage='loading local training snapshot')
            except NameError:
                pass
            return read_local_training_snapshot(snapshot_path)
    print(f"⚠️ {league} no local snapshot found; falling back to network")
    try:
        mark_refresh_state(league, refresh_stage='fetching training data from network')
    except NameError:
        pass
    
    seasons = [
        # Last 9 seasons including current
        ("1617", "2016-17", "16"), ("1718", "2017-18", "17"), ("1819", "2018-19", "18"), ("1920", "2019-20", "19"),
        ("2021", "2020-21", "20"), ("2122", "2021-22", "21"), 
        ("2223", "2022-23", "22"), ("2324", "2023-24", "23"), ("2425", "2024-25", "24"), ("2526", "2025-26", "25"),
    ]
    
    def fetch_season(season_code, season_name, season_key):
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
        df = read_csv_with_timeout(url, timeout=5)
        df['Season'] = season_name
        df['SeasonKey'] = season_key
        df['Weight'] = SEASON_WEIGHTS.get(season_key, 1.0)
        return season_name, df

    all_data = []
    for season in seasons:
        result = {}

        def run_fetch():
            try:
                result['season_name'], result['df'] = fetch_season(*season)
            except Exception as e:
                result['error'] = e

        thread = threading.Thread(target=run_fetch, daemon=True)
        thread.start()
        thread.join(6)
        if thread.is_alive():
            print(f"✗ {league} {season[1]}: timed out")
            continue
        if 'df' in result:
            all_data.append(result['df'])
            print(f"✓ {league} {result['season_name']}")
        else:
            print(f"✗ {league} {season[1]}: {result.get('error')}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Normalize team names
        name_map = {
            "Nott'm Forest": "Nottingham Forest",
            "Nottingham": "Nottingham Forest",
            "Leeds United": "Leeds",
            "Sunderland AFC": "Sunderland",
            "Burnley FC": "Burnley",
            "Brighton & Hove Albion": "Brighton",
            "Manchester City": "Man City",
            "Manchester United": "Man United",
            "Newcastle United": "Newcastle",
            "Tottenham Hotspur": "Tottenham",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves",
            "Atlético Madrid": "Atletico Madrid",
            "Deportivo Alavés": "Alaves",
            "Real Betis": "Betis",
            "Rayo Vallecano": "Rayo Vallecano",
            "AC Milan": "Milan",
            "Hellas Verona": "Verona",
            "1. FC Heidenheim": "Heidenheim",
            "1. FC Köln": "Koln",
            "FC Augsburg": "Augsburg",
            "FC St. Pauli": "St Pauli",
            "Hamburger SV": "Hamburg",
            "SC Freiburg": "Freiburg",
            "TSG Hoffenheim": "Hoffenheim",
            "VfB Stuttgart": "Stuttgart",
            "VfL Wolfsburg": "Wolfsburg",
            "Bayer Leverkusen": "Leverkusen",
            "Borussia Dortmund": "Dortmund",
            "Borussia Mönchengladbach": "Monchengladbach",
            "Paris Saint-Germain": "Paris SG",
        }
        combined['HomeTeam'] = combined['HomeTeam'].replace(name_map)
        combined['AwayTeam'] = combined['AwayTeam'].replace(name_map)
        
        print(f"Total {league}: {len(combined)} matches")
        return combined
    return None


def calculate_team_stats(df, teams):
    """Calculate team statistics"""
    team_stats = {}
    
    for team in teams:
        home_matches = df[df['HomeTeam'] == team].sort_values('Date', ascending=True)
        away_matches = df[df['AwayTeam'] == team].sort_values('Date', ascending=True)
        
        if len(home_matches) < 3:
            continue
        
        home_weights = home_matches['Weight'].values
        away_weights = away_matches['Weight'].values
        
        home_gs = np.average(home_matches['FTHG'].values, weights=home_weights) if len(home_matches) > 0 else 1.4
        home_gc = np.average(home_matches['FTAG'].values, weights=home_weights) if len(home_matches) > 0 else 1.4
        away_gs = np.average(away_matches['FTAG'].values, weights=away_weights) if len(away_matches) > 0 else 1.1
        away_gc = np.average(away_matches['FTHG'].values, weights=away_weights) if len(away_matches) > 0 else 1.4
        
        home_wins = (home_matches['FTR'] == 'H').sum() / max(len(home_matches), 1)
        away_wins = (away_matches['FTR'] == 'A').sum() / max(len(away_matches), 1)
        
        # Recent form
        all_matches = pd.concat([home_matches, away_matches]).sort_values('Date', ascending=True).tail(10)
        recent_weights = np.linspace(1, 2, len(all_matches)) if len(all_matches) > 0 else np.array([1])
        
        form_points = 0
        for i, (_, match) in enumerate(all_matches.iterrows()):
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    form_points += 3 * recent_weights[i]
                elif match['FTR'] == 'D':
                    form_points += 1 * recent_weights[i]
            else:
                if match['FTR'] == 'A':
                    form_points += 3 * recent_weights[i]
                elif match['FTR'] == 'D':
                    form_points += 1 * recent_weights[i]
        
        # Goals in last 5
        recent5 = all_matches.tail(5)
        goals_last5 = sum(m['FTHG'] if m['HomeTeam'] == team else m['FTAG'] for _, m in recent5.iterrows())
        
        # Clean sheets
        home_cs = (home_matches['FTAG'] == 0).sum() / max(len(home_matches), 1)
        away_cs = (away_matches['FTHG'] == 0).sum() / max(len(away_matches), 1)
        
        team_stats[team] = {
            'home_gs': home_gs, 'home_gc': home_gc,
            'away_gs': away_gs, 'away_gc': away_gc,
            'home_win_rate': home_wins, 'away_win_rate': away_wins,
            'form_points': form_points,
            'goals_last5': goals_last5,
            'home_cs_rate': home_cs, 'away_cs_rate': away_cs,
            'matches_played': len(home_matches) + len(away_matches)
        }
    
    return team_stats


def get_head_to_head(df, team1, team2, limit=5):
    """Get head-to-head"""
    h2h = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
             ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))].tail(limit)
    
    if len(h2h) == 0:
        return {'team1_wins': 0, 'team2_wins': 0, 'draws': 0, 'avg_goals': 0, 'matches': []}
    
    team1_wins = team2_wins = draws = total_goals = 0
    matches = []
    
    for _, match in h2h.iterrows():
        if match['HomeTeam'] == team1:
            if match['FTR'] == 'H': team1_wins += 1
            elif match['FTR'] == 'A': team2_wins += 1
            else: draws += 1
            matches.append({'home': team1, 'away': team2, 'score': f"{match['FTHG']}-{match['FTAG']}"})
        else:
            if match['FTR'] == 'A': team1_wins += 1
            elif match['FTR'] == 'H': team2_wins += 1
            else: draws += 1
            matches.append({'home': team2, 'away': team1, 'score': f"{match['FTAG']}-{match['FTHG']}"})
        total_goals += match['FTHG'] + match['FTAG']
    
    return {
        'team1_wins': team1_wins, 'team2_wins': team2_wins, 'draws': draws,
        'avg_goals': total_goals / len(h2h), 'matches': matches
    }


def simulate_season(model, teams, n_sim=100):
    """Simulate season for standings - simplified for speed"""
    standings = {t: {'points': 0, 'gd': 0, 'gf': 0} for t in teams}
    
    # Only simulate a subset for speed
    teams_subset = teams[:8]
    
    for _ in range(n_sim):
        for home in teams_subset:
            for away in teams_subset:
                if home == away:
                    continue
                result = model.predict(home, away)
                if result:
                    r = np.random.random()
                    if r < result['home_prob']:
                        standings[home]['points'] += 3
                    elif r < result['home_prob'] + result['away_prob']:
                        standings[away]['points'] += 3
                    else:
                        standings[home]['points'] += 1
                        standings[away]['points'] += 1
                    standings[home]['gf'] += result['home_goals']
                    standings[home]['gd'] += result['home_goals'] - result['away_goals']
                    standings[away]['gf'] += result['away_goals']
                    standings[away]['gd'] += result['away_goals'] - result['home_goals']
    
    for t in standings:
        standings[t]['points'] /= n_sim
        standings[t]['gd'] /= n_sim
        standings[t]['gf'] /= n_sim
    
    return sorted(standings.items(), key=lambda x: (-x[1]['points'], -x[1]['gd']))


STANDINGS_SIMULATIONS = 120

# Cache
_cache = {}
_cache_time = {}
_cache_locks = {league: threading.Lock() for league in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]}
_refresh_state = {
    league: {'refreshing': False, 'refresh_started_at': None, 'refresh_error': None}
    for league in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
}
_refresh_state_lock = threading.Lock()
DAILY_REFRESH_HOUR_UTC = 2
LEAGUE_REFRESH_OFFSETS = {
    "Premier League": 0,
    "La Liga": 1,
    "Serie A": 2,
    "Bundesliga": 3,
    "Ligue 1": 4,
}


def _scheduled_refresh_time(now, league):
    base_hour = (DAILY_REFRESH_HOUR_UTC + LEAGUE_REFRESH_OFFSETS.get(league, 0)) % 24
    scheduled = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    if now < scheduled:
        scheduled -= timedelta(days=1)
    return scheduled


def _needs_refresh(now, league):
    if league not in _cache or league not in _cache_time:
        return True
    last_refresh = _cache_time[league]
    return last_refresh < _scheduled_refresh_time(now, league)


def latest_match_date(df):
    if df is None or df.empty or 'Date' not in df.columns:
        return None
    completed = df[df.get('FTR').notna()].copy() if 'FTR' in df.columns else df.copy()
    if completed.empty:
        return None
    parsed = pd.to_datetime(completed['Date'], dayfirst=True, errors='coerce')
    latest = parsed.max()
    if pd.isna(latest):
        return None
    return latest.strftime('%Y-%m-%d')


def cache_status_payload(league):
    cache_data = _cache.get(league)
    cache_time = _cache_time.get(league)
    with _refresh_state_lock:
        state = dict(_refresh_state.get(league, {}))
    return {
        'loaded': cache_data is not None,
        'model_type': 'Enhanced Poisson',
        'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None,
        'data_latest_match_date': cache_data.get('data_latest_match_date') or latest_match_date(cache_data['df']) if cache_data else None,
        'matches': cache_data.get('match_count', len(cache_data['df'])) if cache_data else 0,
        'teams': len(cache_data['teams']) if cache_data else 0,
        'league': league,
        'refreshing': bool(state.get('refreshing')),
        'refresh_started_at': state.get('refresh_started_at'),
        'refresh_error': state.get('refresh_error'),
        'refresh_stage': state.get('refresh_stage'),
        'app_version': APP_VERSION,
    }


def mark_refresh_state(league, **updates):
    with _refresh_state_lock:
        state = _refresh_state.setdefault(league, {'refreshing': False, 'refresh_started_at': None, 'refresh_error': None})
        state.update(updates)


def refresh_worker(league, force_refresh=True):
    mark_refresh_state(
        league,
        refreshing=True,
        refresh_started_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        refresh_error=None,
    )
    try:
        cache_data, _ = get_cached_data(league, force_refresh=force_refresh)
        if cache_data is None:
            raise RuntimeError('Could not load data')
        mark_refresh_state(league, refreshing=False, refresh_error=None)
    except Exception as e:
        print(f"❌ Refresh failed for {league}: {e}")
        mark_refresh_state(league, refreshing=False, refresh_error=str(e))


def start_background_refresh(league, force_refresh=True):
    if league not in LEAGUE_DATA:
        return False, 'Unknown league'
    with _refresh_state_lock:
        state = _refresh_state.setdefault(league, {'refreshing': False, 'refresh_started_at': None, 'refresh_error': None})
        if state.get('refreshing'):
            return False, None
        state['refreshing'] = True
        state['refresh_started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        state['refresh_error'] = None
        state['refresh_stage'] = 'queued'
    thread = threading.Thread(target=refresh_worker, args=(league, force_refresh), daemon=True)
    thread.start()
    return True, None


def _daily_refresh_loop():
    # Let startup preloading run first. On small Render instances, kicking off
    # every league refresh immediately can starve the first usable cache load.
    threading.Event().wait(10 * 60)
    while True:
        try:
            now = datetime.now()
            for league in LEAGUE_DATA.keys():
                if _needs_refresh(now, league):
                    start_background_refresh(league, force_refresh=True)
        except Exception as e:
            print(f"❌ Daily refresh loop failed: {e}")
        threading.Event().wait(15 * 60)


def simulate_remaining_season_standings(model, current_df, teams, n_sim=STANDINGS_SIMULATIONS, baseline_df=None):
    actual = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
    played = set()

    source_df = baseline_df if baseline_df is not None and not baseline_df.empty else current_df

    for _, row in source_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in actual or a not in actual:
            continue
        try:
            hg, ag, ftr = int(row['FTHG']), int(row['FTAG']), row['FTR']
        except (ValueError, KeyError, TypeError):
            continue
        played.add((h, a))
        actual[h]['gf'] += hg
        actual[a]['gf'] += ag
        actual[h]['gd'] += hg - ag
        actual[a]['gd'] += ag - hg
        if ftr == 'H':
            actual[h]['pts'] += 3
        elif ftr == 'A':
            actual[a]['pts'] += 3
        else:
            actual[h]['pts'] += 1
            actual[a]['pts'] += 1

    remaining = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    totals = defaultdict(lambda: {'pts': 0.0, 'gd': 0.0, 'gf': 0.0})

    for _ in range(n_sim):
        sim = {t: dict(actual[t]) for t in teams}
        for home, away in remaining:
            result = model.predict(home, away)
            if not result:
                continue

            r = np.random.random()
            if r < result['home_prob']:
                home_goals = max(result['home_goals'], result['away_goals'] + 1)
                away_goals = result['away_goals']
                sim[home]['pts'] += 3
            elif r < result['home_prob'] + result['draw_prob']:
                home_goals = away_goals = max(0, round((result['home_goals'] + result['away_goals']) / 2))
                sim[home]['pts'] += 1
                sim[away]['pts'] += 1
            else:
                away_goals = max(result['away_goals'], result['home_goals'] + 1)
                home_goals = result['home_goals']
                sim[away]['pts'] += 3

            sim[home]['gf'] += home_goals
            sim[away]['gf'] += away_goals
            sim[home]['gd'] += home_goals - away_goals
            sim[away]['gd'] += away_goals - home_goals

        for team in teams:
            totals[team]['pts'] += sim[team]['pts']
            totals[team]['gd'] += sim[team]['gd']
            totals[team]['gf'] += sim[team]['gf']

    standings = [
        {
            'team': team,
            'points': round(totals[team]['pts'] / n_sim, 1),
            'gd': round(totals[team]['gd'] / n_sim, 1),
            'gf': round(totals[team]['gf'] / n_sim, 1),
        }
        for team in teams
    ]
    return sorted(standings, key=lambda x: (-x['points'], -x['gd'], -x['gf']))


def get_cached_data(league, force_refresh=False):
    now = datetime.now()

    # Fast path: cache is fresh, no lock needed
    if not force_refresh and not _needs_refresh(now, league):
        return _cache[league], _cache_time[league]

    lock = _cache_locks.get(league, threading.Lock())
    with lock:
        # Re-check inside lock — another thread may have loaded it while we waited
        now = datetime.now()
        if not force_refresh and not _needs_refresh(now, league):
            return _cache[league], _cache_time[league]

        teams = LEAGUE_DATA[league]["teams"]
        mark_refresh_state(league, refresh_stage='loading precomputed model')
        precomputed = load_precomputed_model(league, teams)
        if precomputed:
            _cache[league] = precomputed
            _cache_time[league] = now
            mark_refresh_state(league, refresh_stage='complete')
            return precomputed, now

        mark_refresh_state(league, refresh_stage='fetching training data')
        df = fetch_data(league)
        if df is None:
            return None, None

        available_teams = list(teams)

        mark_refresh_state(league, refresh_stage='fitting model')
        model = fit_fast_model(df, available_teams)

        loaded_at = datetime.now()
        previous_data = _cache.get(league)
        # Publish the refreshed model as soon as it is usable. Standings are
        # slower to simulate, especially on Render free instances, and should
        # not keep predictions/status stuck in an unloaded state.
        data = {
            'model': model,
            'df': df,
            'teams': available_teams,
            'team_stats': previous_data.get('team_stats', {}) if previous_data else {},
            'standings': previous_data.get('standings', []) if previous_data else [],
        }
        _cache[league] = data
        _cache_time[league] = loaded_at
        mark_refresh_state(league, refresh_stage='complete')

        return data, loaded_at


@app.route('/')
def index():
    return render_template('index.html', 
                          teams=premier_league_teams, 
                          leagues=list(LEAGUE_DATA.keys()),
                          team_colors=TEAM_COLORS)


@app.route('/api/teams')
def get_teams():
    league = request.args.get('league', 'Premier League')
    data = _cache.get(league)
    if data:
        return jsonify({'teams': data['teams'], 'league': league})
    # Still loading — return static fallback so UI isn't blocked
    fallback = LEAGUE_DATA.get(league, LEAGUE_DATA['Premier League'])['teams']
    return jsonify({'teams': list(fallback), 'league': league, 'loading': True})


@app.route('/api/league/<league_name>')
def get_league_info(league_name):
    if league_name in LEAGUE_DATA:
        return jsonify(LEAGUE_DATA[league_name])
    return jsonify({'error': 'League not found'}), 404


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    home = data.get('home_team')
    away = data.get('away_team')
    league = data.get('league', 'Premier League')
    exclude_draw = data.get('exclude_draw', False)
    min_confidence = float(data.get('min_confidence', 0))
    
    if not home or not away:
        return jsonify({'error': 'Please select both teams'}), 400
    
    if home == away:
        return jsonify({'error': 'Teams must be different'}), 400

    cache_data = _cache.get(league)
    cache_time = _cache_time.get(league)

    if cache_data is None:
        return jsonify({'error': 'Data is still loading, please try again in a moment.'}), 503
    
    model = cache_data['model']
    df = cache_data['df']
    team_stats = cache_data['team_stats']
    
    try:
        result = model.predict(home, away, exclude_draw=exclude_draw)
        
        if result is None:
            return jsonify({'error': 'Insufficient data'}), 400
        
        h2h = get_head_to_head(df, home, away)
        home_stats = team_stats.get(home, {})
        away_stats = team_stats.get(away, {})
        
        home_form = "🔥" if home_stats.get('form_points', 0) > 20 else "📉" if home_stats.get('form_points', 0) < 10 else "➡️"
        away_form = "🔥" if away_stats.get('form_points', 0) > 20 else "📉" if away_stats.get('form_points', 0) < 10 else "➡️"
        
        home_goals = result['home_goals']
        away_goals = result['away_goals']
        
        if home_goals > away_goals:
            winner, result_type = home, "Home Win"
        elif away_goals > home_goals:
            winner, result_type = away, "Away Win"
        else:
            winner, result_type = "Draw", "Draw"
        
        if result['confidence'] < min_confidence:
            return jsonify({'error': f'Confidence {result["confidence"]*100:.0f}% below {min_confidence*100:.0f}%'}), 400
        
        return jsonify({
            'home_team': home, 'away_team': away,
            'predicted_home_goals': home_goals, 'predicted_away_goals': away_goals,
            'predicted_score': f"{home_goals} - {away_goals}",
            'predicted_winner': winner, 'result_type': result_type,
            'confidence': f"{result['confidence'] * 100:.0f}%",
            'home_prob': f"{result['home_prob'] * 100:.0f}%",
            'draw_prob': f"{result['draw_prob'] * 100:.0f}%",
            'away_prob': f"{result['away_prob'] * 100:.0f}%",
            'expected_goals': result.get('expected_goals', {}),
            'head_to_head': h2h,
            'home_stats': {k: round(v, 2) if isinstance(v, float) else v for k, v in home_stats.items()},
            'away_stats': {k: round(v, 2) if isinstance(v, float) else v for k, v in away_stats.items()},
            'home_form': home_form, 'away_form': away_form,
            'league': league, 'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/api/standings')
def get_standings():
    league = request.args.get('league', 'Premier League')
    cache_data = _cache.get(league)
    cache_time = _cache_time.get(league)

    if cache_data is None:
        return jsonify({'standings': [], 'league': league, 'loading': True})
    
    # Handle both old tuple format and new dict format
    standings = cache_data.get('standings', [])
    if standings and isinstance(standings[0], tuple):
        # Old format: [(team, {points, gd, gf}), ...]
        formatted = [{'team': t, **s} for t, s in standings]
    else:
        # New format: [{team, points, gd}, ...]
        formatted = standings
    
    return jsonify({
        'standings': formatted,
        'league': league,
        'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None
    })


@app.route('/api/team/<team>')
def get_team_info(team):
    league = request.args.get('league', 'Premier League')
    cache_data = _cache.get(league)
    if cache_data and team in cache_data['team_stats']:
        return jsonify(cache_data['team_stats'][team])
    return jsonify({'error': 'Team not found'}), 404


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    league = request.json.get('league', 'Premier League') if request.json else 'Premier League'
    started, error = start_background_refresh(league, force_refresh=True)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    payload = cache_status_payload(league)
    payload['success'] = True
    payload['started'] = started
    payload['message'] = 'Refresh started' if started else 'Refresh already running'
    return jsonify(payload), 202 if started else 200


@app.route('/api/status')
def get_status():
    league = request.args.get('league', 'Premier League')
    if league not in LEAGUE_DATA:
        return jsonify({'error': 'Unknown league'}), 404
    return jsonify(cache_status_payload(league))


@app.route('/healthz')
def healthz():
    cache_summary = {}
    for league in LEAGUE_DATA.keys():
        cache_summary[league] = cache_status_payload(league)

    return jsonify({
        'status': 'ok',
        'service': 'matchiq',
        'model_type': 'Enhanced Poisson',
        'cached_leagues': cache_summary,
    }), 200


def _preload_all():
    """Preload all leagues on startup so switching is instant."""
    leagues = list(LEAGUE_DATA.keys())
    for league in leagues:
        try:
            print(f"🔄 Preloading {league}...")
            refresh_worker(league, force_refresh=False)
            print(f"✅ {league} ready")
        except Exception as e:
            print(f"❌ Preload failed for {league}: {e}")

threading.Thread(target=_preload_all, daemon=True).start()
threading.Thread(target=_daily_refresh_loop, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
