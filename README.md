# MLBHitPredictor

Author: Owen Halvorson

🎯 Goal
Design a machine learning model that accurately predicts the number of hits in MLB games, with the ultimate goal of identifying profitable opportunities on sportsbooks offering over/under lines that appear to be poorly optimized.

🧠 Strategy
As a dedicated baseball fan, I’ve noticed that the sport is driven by hot and cold streaks — players often perform in cycles. Sportsbooks tend to generalize over/under lines based on a player’s overall reputation or long-term stats, rather than short-term performance trends.

To capitalize on this, I:

Assign heavier weight to recent performance — a struggling star or a surging underdog may offer value if the lines aren’t updated fast enough.

Use multiple rolling averages to capture form trends across entire rosters.

Pull advanced metrics from sources like Statcast (e.g., exit velocity, launch angle, barrel rate) in addition to standard data from the MLB API.

📊 Model Performance Goals
Target MAE (Mean Absolute Error): < 2 hits
This means the model should, on average, predict team hit totals within 2 of the actual result.

🔮 Deployment Plan
Once a reliable model is built:

Compare model predictions with sportsbook over/under lines.

Assume differences between predictions and lines follow a uniform distribution.

Identify bets where the implied probability of the sportsbook line being correct is low.

Surface these opportunities on a front-end dashboard for easy betting analysis.







