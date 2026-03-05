<div align="center">
  <img src="premier-league-predictor/static/images/matchiqlogo.png" alt="MatchIQ" width="220" />

  <p>AI-powered football match predictions across Europe's top 5 leagues.</p>

  <a href="https://matchiq.onrender.com"><strong>matchiq.onrender.com →</strong></a>

  <br />
  <br />

  ![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)
  ![Flask](https://img.shields.io/badge/flask-3.x-lightgrey?style=flat-square)
  ![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
  ![Status](https://img.shields.io/badge/status-live-brightgreen?style=flat-square)
</div>

---

MatchIQ uses the **Dixon-Coles model** — a well-established Poisson regression framework for football prediction — trained on years of historical match data. Select any two teams from the Premier League, La Liga, Serie A, Bundesliga, or Ligue 1, and get a scoreline prediction along with a projected season standings table.

No paid APIs. No accounts. No nonsense.

## Features

- **5 leagues** — Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- **Dixon-Coles model** with attack/defence strength ratings per team
- **Predicted standings** — full projected table for the current season
- **Confidence scores** — transparency about prediction certainty
- **Live data** — fetches and retrains on the latest available match data at startup
- **Light/dark mode** — respects your system preference

## How it works

The Dixon-Coles model treats goals as Poisson-distributed, estimating each team's attacking and defensive strength relative to the league average. A low-score correction factor is applied (the original Dixon-Coles fix for 0-0 and 1-0 results being statistically underweighted by pure Poisson). On each startup, the app:

1. Downloads historical CSVs from [football-data.co.uk](https://www.football-data.co.uk)
2. Fits the model via maximum likelihood estimation
3. Serves predictions through a simple Flask API

Accuracy sits around **55-60% correct result** (Win/Draw/Loss) — roughly in line with academic benchmarks for this model class.

## Running locally

```bash
git clone https://github.com/Frisbiz/MatchIQ.git
cd MatchIQ/premier-league-predictor

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open [localhost:5000](http://localhost:5000). The first load takes 30-60 seconds while data downloads and the model trains.

## Project structure

```
premier-league-predictor/
├── app.py              # Flask app, Dixon-Coles model, prediction logic
├── requirements.txt
├── render.yaml         # Render deployment config
├── Procfile
├── static/
│   └── images/         # Logos and favicon
└── templates/
    └── index.html      # Single-page frontend
```

## Deployment

The app is deployed on Render's free tier. The `render.yaml` in this repo is all you need — fork it, connect your repo on [render.com](https://render.com), and it'll deploy automatically.

One caveat with the free tier: the service spins down after 15 minutes of inactivity, so the first request after idle will be slow. Upgrade to a paid instance if you want it always-on.

## Limitations worth knowing

- Predictions don't account for injuries, suspensions, or squad rotation
- Data updates are limited to what football-data.co.uk publishes (usually a day or two behind)
- The model has no concept of cup fatigue, European games, or managerial changes
- Dixon-Coles is a solid baseline but not state-of-the-art — don't bet money on this

## Data

All match data sourced from [football-data.co.uk](https://www.football-data.co.uk) — a free, well-maintained dataset that's been around since the early 2000s.

## License

MIT. Do whatever you want with it.

---

<div align="center">
  <sub>Built with Flask, scikit-learn, and too much football data.</sub>
</div>
