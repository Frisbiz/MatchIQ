from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)

# Premier League teams 2024-25
premier_league_teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
]

def fetch_extended_data():
    """Fetch 10+ seasons of Premier League data"""
    seasons = [
        ("1415", "2014-15"),
        ("1516", "2015-16"),
        ("1617", "2016-17"),
        ("1718", "2017-18"),
        ("1819", "2018-19"),
        ("1920", "2019-20"),
        ("2021", "2020-21"),
        ("2122", "2021-22"),
        ("2223", "2022-23"),
        ("2324", "2023-24"),
        ("2425", "2024-25"),
    ]
    
    all_data = []
    for season_code, season_name in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        try:
            df = pd.read_csv(url)
            df['Season'] = season_name
            all_data.append(df)
            print(f"✓ Loaded {season_name}")
        except Exception as e:
            print(f"✗ Could not load {season_name}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal matches loaded: {len(combined)}")
        return combined
    return None

def calculate_advanced_stats(df):
    """Calculate comprehensive team statistics"""
    df = df[df['FTR'].notna()].copy()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    team_stats = {}
    
    for team in teams:
        # All matches
        home_matches = df[df['HomeTeam'] == team].sort_values('Date', ascending=True)
        away_matches = df[df['AwayTeam'] == team].sort_values('Date', ascending=True)
        all_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date', ascending=True)
        
        if len(all_matches) < 5:
            continue
        
        # Basic stats
        home_gs = home_matches['FTHG'].mean() if len(home_matches) > 0 else 1.5
        home_gc = home_matches['FTAG'].mean() if len(home_matches) > 0 else 1.5
        away_gs = away_matches['FTAG'].mean() if len(away_matches) > 0 else 1.2
        away_gc = away_matches['FTHG'].mean() if len(away_matches) > 0 else 1.2
        
        # Win rates
        home_wins = len(home_matches[home_matches['FTR'] == 'H']) / max(len(home_matches), 1)
        away_wins = len(away_matches[away_matches['FTR'] == 'A']) / max(len(away_matches), 1)
        total_matches = len(all_matches)
        overall_win_rate = (len(home_matches[home_matches['FTR'] == 'H']) + 
                           len(away_matches[away_matches['FTR'] == 'A'])) / max(total_matches, 1)
        
        # Recent form (last 10 matches)
        recent = all_matches.tail(10)
        form_points = 0
        for _, match in recent.iterrows():
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
            else:
                if match['FTR'] == 'A':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
        
        # Goals in last 5
        recent5 = all_matches.tail(5)
        goals_last5 = 0
        for _, match in recent5.iterrows():
            if match['HomeTeam'] == team:
                goals_last5 += match['FTHG']
            else:
                goals_last5 += match['FTAG']
        
        # xG (expected goals) if available
        home_xg = home_matches['HxG'].mean() if 'HxG' in home_matches.columns and home_matches['HxG'].notna().any() else home_gs
        away_xg = away_matches['AxG'].mean() if 'AxG' in away_matches.columns and away_matches['AxG'].notna().any() else away_gs
        
        # Clean sheets
        home_cs = len(home_matches[home_matches['FTAG'] == 0]) / max(len(home_matches), 1)
        away_cs = len(away_matches[away_matches['FTHG'] == 0]) / max(len(away_matches), 1)
        
        team_stats[team] = {
            'home_gs': home_gs,
            'home_gc': home_gc,
            'away_gs': away_gs,
            'away_gc': away_gc,
            'home_win_rate': home_wins,
            'away_win_rate': away_wins,
            'overall_win_rate': overall_win_rate,
            'form_points': form_points,
            'goals_last5': goals_last5,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_cs_rate': home_cs,
            'away_cs_rate': away_cs,
            'matches_played': total_matches
        }
    
    return team_stats

def get_head_to_head(df, team1, team2):
    """Get head-to-head statistics between two teams"""
    h2h = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
             ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]
    
    if len(h2h) == 0:
        return {'team1_wins': 0, 'team2_wins': 0, 'draws': 0, 'avg_goals': 2.5}
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    total_goals = 0
    
    for _, match in h2h.iterrows():
        if match['HomeTeam'] == team1:
            if match['FTR'] == 'H':
                team1_wins += 1
            elif match['FTR'] == 'A':
                team2_wins += 1
            else:
                draws += 1
        else:
            if match['FTR'] == 'A':
                team1_wins += 1
            elif match['FTR'] == 'H':
                team2_wins += 1
            else:
                draws += 1
        total_goals += match['FTHG'] + match['FTAG']
    
    return {
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'avg_goals': total_goals / len(h2h)
    }

def build_features(df, team_stats):
    """Build feature matrix for training"""
    X = []
    y_home = []
    y_away = []
    
    for _, match in df.iterrows():
        home = match['HomeTeam']
        away = match['AwayTeam']
        
        if home not in team_stats or away not in team_stats:
            continue
        
        hs = team_stats[home]
        ast = team_stats[away]
        h2h = get_head_to_head(df, home, away)
        
        features = [
            hs['home_gs'], hs['home_gc'], hs['home_win_rate'],
            hs['form_points'] / 10.0,  # Normalize
            hs['goals_last5'] / 5.0,
            hs['home_xg'], hs['home_cs_rate'],
            ast['away_gs'], ast['away_gc'], ast['away_win_rate'],
            ast['form_points'] / 10.0,
            ast['goals_last5'] / 5.0,
            ast['away_xg'], ast['away_cs_rate'],
            h2h['team1_wins'] / max(h2h['team1_wins'] + h2h['team2_wins'] + h2h['draws'], 1),
            h2h['avg_goals'] / 5.0,
            hs['overall_win_rate'],
            ast['overall_win_rate']
        ]
        
        X.append(features)
        y_home.append(match['FTHG'])
        y_away.append(match['FTAG'])
    
    return np.array(X), np.array(y_home), np.array(y_away)

def train_models(df, team_stats):
    """Train prediction models"""
    print("Building feature matrix...")
    X, y_home, y_away = build_features(df, team_stats)
    
    print(f"Training on {len(X)} matches...")
    
    # Use Gradient Boosting for better accuracy
    model_home = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model_away = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)
    
    # Calculate accuracy
    home_score = model_home.score(X, y_home)
    away_score = model_away.score(X, y_away)
    print(f"Model R² scores - Home: {home_score:.3f}, Away: {away_score:.3f}")
    
    return model_home, model_away

def predict_match(home, away, team_stats, models, df):
    """Predict match outcome"""
    if home not in team_stats or away not in team_stats:
        return None, None, 0
    
    hs = team_stats[home]
    ast = team_stats[away]
    h2h = get_head_to_head(df, home, away)
    
    features = np.array([[
        hs['home_gs'], hs['home_gc'], hs['home_win_rate'],
        hs['form_points'] / 10.0,
        hs['goals_last5'] / 5.0,
        hs['home_xg'], hs['home_cs_rate'],
        ast['away_gs'], ast['away_gc'], ast['away_win_rate'],
        ast['form_points'] / 10.0,
        ast['goals_last5'] / 5.0,
        ast['away_xg'], ast['away_cs_rate'],
        h2h['team1_wins'] / max(h2h['team1_wins'] + h2h['team2_wins'] + h2h['draws'], 1),
        h2h['avg_goals'] / 5.0,
        hs['overall_win_rate'],
        ast['overall_win_rate']
    ]])
    
    model_home, model_away = models
    
    home_pred = model_home.predict(features)[0]
    away_pred = model_away.predict(features)[0]
    
    # Round and ensure non-negative
    home_goals = max(0, round(home_pred))
    away_goals = max(0, round(away_pred))
    
    # Calculate confidence based on prediction certainty
    home_std = np.std([tree[0].predict(features)[0] for tree in model_home.estimators_])
    away_std = np.std([tree[0].predict(features)[0] for tree in model_away.estimators_])
    avg_std = (home_std + away_std) / 2
    confidence = max(0.4, min(0.9, 1.0 - (avg_std / 3.0)))
    
    return home_goals, away_goals, confidence

# Global storage
models = None
team_stats = None
df_global = None

@app.before_request
def init_model():
    """Lazy load model on first request"""
    global models, team_stats, df_global
    
    if models is None:
        print("🔄 Initializing prediction models...")
        df_global = fetch_extended_data()
        
        if df_global is not None and len(df_global) > 100:
            team_stats = calculate_advanced_stats(df_global)
            models = train_models(df_global, team_stats)
            print(f"✅ Model ready! Using {len(df_global)} matches from {len(team_stats)} teams")
        else:
            print("⚠️ Could not load sufficient data")

@app.route('/')
def index():
    return render_template('index.html', teams=premier_league_teams)

@app.route('/api/teams')
def get_teams():
    return jsonify({'teams': premier_league_teams})

@app.route('/api/predict', methods=['POST'])
def predict():
    global models, team_stats, df_global
    
    data = request.json
    home = data.get('home_team')
    away = data.get('away_team')
    
    if not home or not away:
        return jsonify({'error': 'Please select both teams'}), 400
    
    if home == away:
        return jsonify({'error': 'Teams must be different'}), 400
    
    if models is None or team_stats is None:
        return jsonify({'error': 'Model not ready. Please try again in a moment.'}), 503
    
    try:
        home_goals, away_goals, confidence = predict_match(home, away, team_stats, models, df_global)
        
        if home_goals is None:
            return jsonify({'error': 'Insufficient data for these teams'}), 400
        
        # Determine result
        if home_goals > away_goals:
            winner = home
            result = "Home Win"
        elif away_goals > home_goals:
            winner = away
            result = "Away Win"
        else:
            winner = "Draw"
            result = "Draw"
        
        # Get team stats for display
        hs = team_stats.get(home, {})
        ast = team_stats.get(away, {})
        
        return jsonify({
            'home_team': home,
            'away_team': away,
            'predicted_home_goals': int(home_goals),
            'predicted_away_goals': int(away_goals),
            'predicted_score': f"{int(home_goals)} - {int(away_goals)}",
            'predicted_winner': winner,
            'result_type': result,
            'confidence': f"{confidence * 100:.0f}%",
            'home_stats': {
                'avg_goals_scored': round(hs.get('home_gs', 0), 2),
                'avg_goals_conceded': round(hs.get('home_gc', 0), 2),
                'win_rate': f"{hs.get('home_win_rate', 0) * 100:.0f}%"
            },
            'away_stats': {
                'avg_goals_scored': round(ast.get('away_gs', 0), 2),
                'avg_goals_conceded': round(ast.get('away_gc', 0), 2),
                'win_rate': f"{ast.get('away_win_rate', 0) * 100:.0f}%"
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

@app.route('/api/stats/<team>')
def get_team_stats(team):
    if team_stats and team in team_stats:
        return jsonify(team_stats[team])
    return jsonify({'error': 'Team not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
