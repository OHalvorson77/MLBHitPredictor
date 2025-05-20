import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import joblib





df = pd.read_csv("processed_hits_dataset7.csv")



feature_cols = [
    'lineup_avg', 'lineup_obp', 'lineup_slg',
    'starter_era', 'starter_whip',
    'bullpen_era', 'bullpen_whip',
    'is_home', 'bvp_avg', 'bvp_ops', 'bvp_hr', 'lineup_avg_exit_velocity', 'lineup_launch_angle', 'lineup_xba', 'lineup_hard_hit_rate', 'lineup_chase','lineup_whiff', 'bullpen_righty_pct',
    'avg_exit_velocity', 'avg_launch_angle', 'xba', 'hard_hit_rate', 'whiff', 'chase', 'strikeout_rate', 'walk_rate', 'bullpen_bvp_avg', 'bullpen_bvp_ops', 'bull_avg_exit_velocity',
    'bull_avg_launch_angle', 'bull_xba', 'bull_hard_hit_rate', 'bull_whiff', 'bull_chase','bull_strikeout_rate', 'bull_walk_rate', 'x_b', 'y_b', 'x_s', 'y_s', 'x_l', 'y_l', 'altitude', 'stadium_factor', 'left', 'center', 'right'
]
X = df[feature_cols]
y = df['runs']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
joblib.dump(best_model, 'best_random_forest_model.pkl')
print("Model saved as best_random_forest_model.pkl")

