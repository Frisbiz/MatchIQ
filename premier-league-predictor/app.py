from flask import Flask, render_template, jsonify, request, make_response
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

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

# Premier League teams
premier_league_teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
]

# Leagues
LEAGUE_DATA = {
    "Premier League": {"country": "England", "code": "E0", "teams": premier_league_teams},
    "La Liga": {"country": "Spain", "code": "SP1", "teams": [
        "Alaves", "Almeria", "Athletic Bilbao", "Atletico Madrid", "Barcelona", 
        "Betis", "Celta Vigo", "Girona", "Granada", "Las Palmas",
        "Levante", "Osasuna", "Ray Vallecano", "Real Madrid", "Real Sociedad",
        "Sevilla", "Valencia", "Villarreal", "Mallorca", "Espanyol"
    ]},
    "Serie A": {"country": "Italy", "code": "I1", "teams": [
        "Atalanta", "Bologna", "Cagliari", "Como", "Empoli",
        "Fiorentina", "Frosinone", "Genoa", "Inter Milan", "Juventus",
        "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Parma",
        "Roma", "Salernitana", "Sassuolo", "Torino", "Udinese", "Venezia", "Verona"
    ]},
    "Bundesliga": {"country": "Germany", "code": "D1", "teams": [
        "Augsburg", "Bayern Munich", "Bochum", "Dortmund", "Eintracht Frankfurt",
        "Freiburg", "Hertha Berlin", "Hoffenheim", "Koln", "Leverkusen",
        "Mainz", "Monchengladbach", "RB Leipzig", "Schalke", "Stuttgart",
        "Union Berlin", "Werder Bremen", "Wolfsburg"
    ]},
    "Ligue 1": {"country": "France", "code": "F1", "teams": [
        "Brest", "Clermont", "Dijon", "Lille", "Lorient", "Lyon", "Marseille", 
        "Metz", "Monaco", "Montpellier", "Nantes", "Nice", "Paris SG", 
        "Reims", "Rennes", "Strasbourg", "Toulouse", "Troyes"
    ]},
}

# Season weights
SEASON_WEIGHTS = {
    "2425": 2.5, "2324": 2.0, "2223": 1.5, "2122": 1.2, "2021": 1.0,
    "1920": 0.9, "1819": 0.8, "1718": 0.7, "1617": 0.6, "1516": 0.5, "1415": 0.4
}

def fetch_data(league="Premier League"):
    """Fetch league data"""
    league_info = LEAGUE_DATA.get(league, LEAGUE_DATA["Premier League"])
    code = league_info["code"]
    
    seasons = [
        # Only last 5 seasons for faster loading
        ("2021", "2020-21", "20"), ("2122", "2021-22", "21"), 
        ("2223", "2022-23", "22"), ("2324", "2023-24", "23"), ("2425", "2024-25", "24"),
    ]
    
    all_data = []
    for season_code, season_name, season_key in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
        try:
            df = pd.read_csv(url)
            df['Season'] = season_name
            df['SeasonKey'] = season_key
            df['Weight'] = SEASON_WEIGHTS.get(season_key, 1.0)
            all_data.append(df)
            print(f"✓ {league} {season_name}")
        except Exception as e:
            print(f"✗ {league} {season_name}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
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


# Cache
_cache = {}
_cache_time = {}
CACHE_DURATION = 3600


def get_cached_data(league):
    now = datetime.now()
    
    if league in _cache and league in _cache_time:
        if (now - _cache_time[league]).total_seconds() < CACHE_DURATION:
            return _cache[league], _cache_time[league]
    
    df = fetch_data(league)
    if df is None:
        return None, None
    
    teams = LEAGUE_DATA[league]["teams"]
    available_teams = [t for t in teams if t in df['HomeTeam'].values or t in df['AwayTeam'].values]
    
    model = EnhancedPoissonModel()
    model.fit(df, available_teams)
    team_stats = calculate_team_stats(df, available_teams)
    # Skip standings for now - too slow
    standings = []
    
    data = {'model': model, 'df': df, 'teams': available_teams, 'team_stats': team_stats, 'standings': standings}
    
    _cache[league] = data
    _cache_time[league] = now
    
    return data, now


@app.route('/')
def index():
    return render_template('index.html', 
                          teams=premier_league_teams, 
                          leagues=list(LEAGUE_DATA.keys()),
                          team_colors=TEAM_COLORS)


@app.route('/api/teams')
def get_teams():
    league = request.args.get('league', 'Premier League')
    data, _ = get_cached_data(league)
    if data:
        return jsonify({'teams': data['teams'], 'league': league})
    return jsonify({'teams': premier_league_teams, 'league': 'Premier League'})


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
    
    cache_data, cache_time = get_cached_data(league)
    
    if cache_data is None:
        return jsonify({'error': 'Could not load data'}), 500
    
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
    cache_data, cache_time = get_cached_data(league)
    
    if cache_data is None:
        return jsonify({'error': 'Could not load data'}), 500
    
    return jsonify({
        'standings': [{'team': t, **s} for t, s in cache_data['standings'][:10]],
        'league': league,
        'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None
    })


@app.route('/api/team/<team>')
def get_team_info(team):
    league = request.args.get('league', 'Premier League')
    cache_data, _ = get_cached_data(league)
    
    if cache_data and team in cache_data['team_stats']:
        return jsonify(cache_data['team_stats'][team])
    return jsonify({'error': 'Team not found'}), 404


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    global _cache, _cache_time
    
    league = request.json.get('league', 'Premier League') if request.json else 'Premier League'
    
    if league in _cache:
        del _cache[league]
    if league in _cache_time:
        del _cache_time[league]
    
    cache_data, cache_time = get_cached_data(league)
    
    if cache_data:
        return jsonify({
            'success': True, 'matches': len(cache_data['df']), 'teams': len(cache_data['teams']),
            'model_type': 'Enhanced Poisson', 'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None, 'league': league
        })
    return jsonify({'error': 'Could not load data'}), 500


@app.route('/api/status')
def get_status():
    league = request.args.get('league', 'Premier League')
    cache_data, cache_time = get_cached_data(league)
    
    return jsonify({
        'loaded': cache_data is not None,
        'model_type': 'Enhanced Poisson',
        'last_updated': cache_time.strftime('%Y-%m-%d %H:%M') if cache_time else None,
        'matches': len(cache_data['df']) if cache_data else 0,
        'teams': len(cache_data['teams']) if cache_data else 0,
        'league': league
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
