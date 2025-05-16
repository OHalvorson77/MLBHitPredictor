# pip install requests pandas scikit-learn pybaseball


import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


SEASON = '2024'
WINDOW_SIZE = 5  # Number of games for rolling average
START_DATE = '2024-08-01'
END_DATE = '2024-09-30'

def get_schedule(start_date, end_date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve schedule: {response.status_code}")
        return []
    data = response.json()
    games = []
    for date_info in data.get('dates', []):
        for game in date_info.get('games', []):
            games.append({
                'gamePk': game['gamePk'],
                'gameDate': game['gameDate'],
                'homeTeam': game['teams']['home']['team']['id'],
                'awayTeam': game['teams']['away']['team']['id']
            })
    return games


def get_boxscore(gamePk):
    url = f"https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve boxscore for game {gamePk}: {response.status_code}")
        return None
    return response.json()


def get_player_game_logs(player_id, season=SEASON, group='hitting'):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={season}&group={group}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve game logs for player {player_id}: {response.status_code}")
        return []
    data = response.json()
    stats = data.get('stats', [])
    if not stats or 'splits' not in stats[0]:
        return []
    return stats[0]['splits']


def compute_weighted_rolling_average(stats_list, stat_key, window=WINDOW_SIZE):
    if len(stats_list) < window:
        return None
    values = []
    for stat in stats_list[-window:]:
        value = stat['stat'].get(stat_key)
        if value is None:
            return None
        try:
            values.append(float(value))
        except ValueError:
            return None
    weights = np.arange(1, window + 1)
    return np.average(values, weights=weights)


def extract_lineup_player_ids(boxscore_team):
    player_ids = []
    for player_key, player_info in boxscore_team['players'].items():
        if player_info.get('position', {}).get('code') != 'P':
            player_ids.append(player_info['person']['id'])
    return player_ids[:9]


def get_team_features(team_id, player_ids):
    avg_list = []
    obp_list = []
    slg_list = []
    vs_rhp_avg = []
    vs_lhp_avg = []
    vs_rhp_ops = []
    vs_lhp_ops = []
    ev_list = []
    la_list = []
    barrel_list = []

    for pid in player_ids:
        logs = get_player_game_logs(pid)
        if not logs:
            continue

        avg = compute_weighted_rolling_average(logs, 'avg')
        obp = compute_weighted_rolling_average(logs, 'obp')
        slg = compute_weighted_rolling_average(logs, 'slg')
        agg = get_player_platoon_splits(pid)
        statcast = compute_statcast_rolling_average(pid, days=60)

        if avg is not None:
            avg_list.append(avg)
        if obp is not None:
            obp_list.append(obp)
        if slg is not None:
            slg_list.append(slg)

        if agg:
            if agg['vs_rhp_avg'] is not None:
                vs_rhp_avg.append(agg['vs_rhp_avg'])
            if agg['vs_lhp_avg'] is not None:
                vs_lhp_avg.append(agg['vs_lhp_avg'])
            if agg['vs_rhp_ops'] is not None:
                vs_rhp_ops.append(agg['vs_rhp_ops'])
            if agg['vs_lhp_ops'] is not None:
                vs_lhp_ops.append(agg['vs_lhp_ops'])

        if statcast:
            if statcast['exit_velocity'] is not None:
                ev_list.append(statcast['exit_velocity'])
            if statcast['launch_angle'] is not None:
                la_list.append(statcast['launch_angle'])
            if statcast['barrel_rate'] is not None:
                barrel_list.append(statcast['barrel_rate'])

    features = {
        'lineup_avg': np.mean(avg_list) if avg_list else None,
        'lineup_obp': np.mean(obp_list) if obp_list else None,
        'lineup_slg': np.mean(slg_list) if slg_list else None,
        'lineup_arhp': np.mean(vs_rhp_avg) if vs_rhp_avg else None,
        'lineup_alhp': np.mean(vs_lhp_avg) if vs_lhp_avg else None,
        'lineup_orhp': np.mean(vs_rhp_ops) if vs_rhp_ops else None,
        'lineup_olhp': np.mean(vs_lhp_ops) if vs_lhp_ops else None,
        'lineup_exit_velocity': np.mean(ev_list) if ev_list else None,
        'lineup_launch_angle': np.mean(la_list) if la_list else None,
        'lineup_barrel_rate': np.mean(barrel_list) if barrel_list else None
    }

    return features


import requests
import time

def get_bvp_stats_for_lineup(batter_ids, pitcher_id):
    bvp_stats = {
        'bvp_avg': [],
        'bvp_ops': [],
        'bvp_hr': []
    }

    for batter_id in batter_ids:
        url = f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=vsPlayer&opposingPlayerId={pitcher_id}"
        response = requests.get(url)
        if response.status_code != 200:
            continue

        data = response.json()
        splits = data.get("stats", [])[0].get("splits", [])
        if splits:
            stat_line = splits[0]['stat']
            try:
                bvp_stats['bvp_avg'].append(float(stat_line.get('avg', 0)))
                bvp_stats['bvp_ops'].append(float(stat_line.get('ops', 0)))
                bvp_stats['bvp_hr'].append(int(stat_line.get('homeRuns', 0)))
            except (ValueError, TypeError):
                continue
        
        time.sleep(0.2)  # Avoid rate-limiting

    
    aggregated = {
        'bvp_avg': round(sum(bvp_stats['bvp_avg']) / len(bvp_stats['bvp_avg']), 3) if bvp_stats['bvp_avg'] else 0.25,
        'bvp_ops': round(sum(bvp_stats['bvp_ops']) / len(bvp_stats['bvp_ops']), 3) if bvp_stats['bvp_ops'] else 0.7,
        'bvp_hr': sum(bvp_stats['bvp_hr']) if bvp_stats['bvp_hr'] else 0
    }

    return aggregated


from pybaseball import statcast_batter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def compute_statcast_rolling_average(player_id, days=60):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    try:
        df = statcast_batter(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), player_id)
        if df.empty:
            return None

        df = df.dropna(subset=["exit_velocity", "launch_angle"])
        avg_ev = df["exit_velocity"].mean()
        avg_la = df["launch_angle"].mean()
        barrel_rate = df["barrel"].sum() / len(df) if len(df) > 0 else 0

        return {
            "exit_velocity": round(avg_ev, 2) if not np.isnan(avg_ev) else None,
            "launch_angle": round(avg_la, 2) if not np.isnan(avg_la) else None,
            "barrel_rate": round(barrel_rate * 100, 2)
        }
    except Exception as e:
        print(f"Statcast error for player {player_id}: {e}")
        return None




def get_player_platoon_splits(player_id):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=statSplits&group=hitting"
    response = requests.get(url)
    if response.status_code != 200:
        return {}
    
    splits = response.json().get("stats", [])[0].get("splits", [])
    vs_rhp, vs_lhp = {}, {}
    for split in splits:
        if split['split']['handedness'] == 'Right':
            vs_rhp = split['stat']
        elif split['split']['handedness'] == 'Left':
            vs_lhp = split['stat']

    return {
        'vs_rhp_avg': float(vs_rhp.get('avg', 0)),
        'vs_lhp_avg': float(vs_lhp.get('avg', 0)),
        'vs_rhp_ops': float(vs_rhp.get('ops', 0)),
        'vs_lhp_ops': float(vs_lhp.get('ops', 0)),
    }



def get_pitching_features(starter_id, bullpen_ids, player_ids):
    
    starter_logs = get_player_game_logs(starter_id, group='pitching')
    starter_era = compute_weighted_rolling_average(starter_logs, 'era')
    starter_whip = compute_weighted_rolling_average(starter_logs, 'whip')

    innings_pitched = []
    for log in starter_logs:
        ip = log.get('ip')
        if ip:
            try:
                parts = str(ip).split('.')
                ip_float = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 else 0)
                innings_pitched.append(ip_float)
            except:
                continue
    avg_ip = round(np.mean(innings_pitched), 2) if innings_pitched else None

    starter_info = requests.get(f"https://statsapi.mlb.com/api/v1/people/{starter_id}").json()
    starter_hand = starter_info.get('people', [{}])[0].get('pitchHand', {}).get('code', None)  # 'R' or 'L'

    aggregated = get_bvp_stats_for_lineup(player_ids, starter_id)
    bvp_avg = aggregated['bvp_avg']
    bvp_ops = aggregated['bvp_ops']
    bvp_hr = aggregated['bvp_hr']

    bullpen_era_list = []
    bullpen_whip_list = []
    right_handed_count = 0

    for pid in bullpen_ids:
        logs = get_player_game_logs(pid, group='pitching')
        if not logs:
            continue

        era = compute_weighted_rolling_average(logs, 'era')
        whip = compute_weighted_rolling_average(logs, 'whip')
        if era is not None:
            bullpen_era_list.append(era)
        if whip is not None:
            bullpen_whip_list.append(whip)

        info = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}").json()
        hand = info.get('people', [{}])[0].get('pitchHand', {}).get('code', None)
        if hand == 'R':
            right_handed_count += 1

    bullpen_era = np.mean(bullpen_era_list) if bullpen_era_list else None
    bullpen_whip = np.mean(bullpen_whip_list) if bullpen_whip_list else None
    bullpen_righty_pct = round(100 * right_handed_count / len(bullpen_ids), 2) if bullpen_ids else None

    return {
        'starter_era': starter_era,
        'starter_whip': starter_whip,
        'starter_avg_ip': avg_ip,
        'starter_hand': starter_hand,  # 'R', 'L', or None
        'bullpen_era': bullpen_era,
        'bullpen_whip': bullpen_whip,
        'bullpen_righty_pct': bullpen_righty_pct,
        'bvp_avg': bvp_avg,
        'bvp_ops': bvp_ops,
        'bvp_hr': bvp_hr
    }



games = get_schedule(START_DATE, END_DATE)
data_rows = []

i=0

for game in games:
    print(i)
    i=i+1
    gamePk = game['gamePk']
    boxscore = get_boxscore(gamePk)
    if boxscore is None:
        continue

    for team_type in ['home', 'away']:
        team_info = boxscore['teams'][team_type]
        opponent_type = 'away' if team_type == 'home' else 'home'
        opponent_info = boxscore['teams'][opponent_type]

        team_id = team_info['team']['id']
        opponent_pitchers = opponent_info['pitchers']
        if not opponent_pitchers:
            continue
            

        starter_id = opponent_pitchers[0]
        bullpen_ids = opponent_pitchers[1:] if len(opponent_pitchers) > 1 else []

        lineup_player_ids = extract_lineup_player_ids(team_info)
        team_features = get_team_features(team_id, lineup_player_ids)
        pitching_features = get_pitching_features(starter_id, bullpen_ids, lineup_player_ids)

        hits = team_info['teamStats']['batting'].get('hits')
        if hits is None:
            continue

        row = {
            'team_id': team_id,
            'is_home': team_type == 'home',
            'hits': hits
        }
        row.update(team_features)
        row.update(pitching_features)
        data_rows.append(row)





df = pd.DataFrame(data_rows)
df.dropna(inplace=True)  

feature_cols = [
    'lineup_avg', 'lineup_obp', 'lineup_slg',
    'starter_era', 'starter_whip',
    'bullpen_era', 'bullpen_whip',
    'is_home', 'bvp_avg', 'bvp_ops', 'bvp_hr','lineup_arhp', 'lineup_alhp', 'lineup_orhp', 'lineup_olhp'
]
X = df[feature_cols]
y = df['hits']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)
rmse =mean_absolute_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")







