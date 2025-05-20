from pybaseball import statcast

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

STADIUM_ATTRIBUTES = {
    "Angel Stadium": {
        "left": 347, "center": 396, "right": 330,
        "altitude": 160, "park_factor": 96
    },
    "Busch Stadium": {
        "left": 336, "center": 400, "right": 335,
        "altitude": 466, "park_factor": 95
    },
    "Chase Field": {
        "left": 330, "center": 407, "right": 335,
        "altitude": 1100, "park_factor": 105
    },
    "Citizens Bank Park": {
        "left": 329, "center": 401, "right": 330,
        "altitude": 30, "park_factor": 104
    },
    "Citi Field": {
        "left": 335, "center": 408, "right": 330,
        "altitude": 20, "park_factor": 97
    },
    "Comerica Park": {
        "left": 345, "center": 420, "right": 330,
        "altitude": 580, "park_factor": 98
    },
    "Coors Field": {
        "left": 347, "center": 415, "right": 350,
        "altitude": 5200, "park_factor": 119
    },
    "Dodger Stadium": {
        "left": 330, "center": 400, "right": 330,
        "altitude": 570, "park_factor": 101
    },
    "Fenway Park": {
        "left": 310, "center": 390, "right": 302,
        "altitude": 20, "park_factor": 108
    },
    "Globe Life Field": {
        "left": 329, "center": 407, "right": 326,
        "altitude": 600, "park_factor": 101
    },
    "Great American Ball Park": {
        "left": 328, "center": 404, "right": 325,
        "altitude": 480, "park_factor": 105
    },
    "Guaranteed Rate Field": {
        "left": 330, "center": 400, "right": 335,
        "altitude": 600, "park_factor": 104
    },
    "Kauffman Stadium": {
        "left": 330, "center": 410, "right": 330,
        "altitude": 750, "park_factor": 98
    },
    "LoanDepot Park": {
        "left": 344, "center": 407, "right": 335,
        "altitude": 10, "park_factor": 93
    },
    "Minute Maid Park": {
        "left": 315, "center": 409, "right": 326,
        "altitude": 50, "park_factor": 102
    },
    "Nationals Park": {
        "left": 336, "center": 402, "right": 335,
        "altitude": 60, "park_factor": 101
    },
    "Oakland Coliseum": {
        "left": 330, "center": 400, "right": 330,
        "altitude": 50, "park_factor": 92
    },
    "Oracle Park": {
        "left": 339, "center": 399, "right": 309,
        "altitude": 10, "park_factor": 94
    },
    "Oriole Park at Camden Yards": {
        "left": 384, "center": 410, "right": 318,
        "altitude": 30, "park_factor": 100
    },
    "Petco Park": {
        "left": 336, "center": 396, "right": 322,
        "altitude": 50, "park_factor": 93
    },
    "PNC Park": {
        "left": 325, "center": 399, "right": 320,
        "altitude": 730, "park_factor": 97
    },
    "Progressive Field": {
        "left": 325, "center": 400, "right": 325,
        "altitude": 653, "park_factor": 100
    },
    "Rogers Centre": {
        "left": 328, "center": 400, "right": 328,
        "altitude": 250, "park_factor": 102
    },
    "T-Mobile Park": {
        "left": 331, "center": 401, "right": 326,
        "altitude": 30, "park_factor": 95
    },
    "Target Field": {
        "left": 339, "center": 403, "right": 328,
        "altitude": 840, "park_factor": 98
    },
    "Tropicana Field": {
        "left": 315, "center": 404, "right": 322,
        "altitude": 3, "park_factor": 97
    },
    "Truist Park": {
        "left": 335, "center": 400, "right": 325,
        "altitude": 1050, "park_factor": 101
    },
    "Wrigley Field": {
        "left": 355, "center": 400, "right": 353,
        "altitude": 594, "park_factor": 104
    },
    "Yankee Stadium": {
        "left": 318, "center": 408, "right": 314,
        "altitude": 55, "park_factor": 105
    },
    "American Family Field": {
        "left": 344, "center": 400, "right": 345,
        "altitude": 640, "park_factor": 102
    },
    "Busch Stadium": {
        "left": 336, "center": 400, "right": 335,
        "altitude": 466, "park_factor": 95
    }
}


# Get data from April 1, 2024 to April 7, 2024
data = statcast(start_dt="2024-06-01", end_dt="2024-08-03")

SEASON = '2024'
WINDOW_SIZE = 5  # Number of games for rolling average
START_DATE = '2024-06-15'
END_DATE = '2024-08-03'


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
                'awayTeam': game['teams']['away']['team']['id'],
                'venue': game.get('venue', {}).get('name')  

            })
    return games


def get_stadium_features(venue_name):
    attrs = STADIUM_ATTRIBUTES.get(venue_name, {})
    return {
        "altitude": attrs.get("altitude", 0),
        "stadium_factor": attrs.get("park_factor", 1.0),
        "cf_distance": attrs.get("center", 400),
        "lf_distance": attrs.get("left", 330),
        "rf_distance": attrs.get("right", 330),
    }



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


def get_team_features(data, team_id, player_ids):
    avg_list = []
    obp_list = []
    slg_list = []
    vs_rhp_avg = []
    vs_lhp_avg = []
    vs_rhp_ops = []
    vs_lhp_ops = []
    hr_list=[]
    avg_exit_velocity = []
    avg_launch_angle = []
    xba = []
    hard_hit_rate = []
    chase = []
    whiff = []
    x_l=[]
    y_l=[]

    for pid in player_ids:
        logs = get_player_game_logs(pid)
        if not logs:
            continue

        avg = compute_weighted_rolling_average(logs, 'avg')
        obp = compute_weighted_rolling_average(logs, 'obp')
        slg = compute_weighted_rolling_average(logs, 'slg')
        hr = compute_weighted_rolling_average(logs, 'hr')


        agg = get_player_platoon_splits(pid)
        advanced_agg=get_avg_batter_stats(data, pid)

        if avg is not None:
            avg_list.append(avg)
        if obp is not None:
            obp_list.append(obp)
        if slg is not None:
            slg_list.append(slg)
        if hr is not None:
            hr_list.append(slg)


        if agg:
            if agg['vs_rhp_avg'] is not None:
                print('True')
                vs_rhp_avg.append(agg['vs_rhp_avg'])
            if agg['vs_lhp_avg'] is not None:
                print('True')
                vs_lhp_avg.append(agg['vs_lhp_avg'])
            if agg['vs_rhp_ops'] is not None:
                vs_rhp_ops.append(agg['vs_rhp_ops'])
            if agg['vs_lhp_ops'] is not None:
                vs_lhp_ops.append(agg['vs_lhp_ops'])
        if advanced_agg:
            if advanced_agg['avg_exit_velocity'] is not None:
                print('True')
                avg_exit_velocity.append(advanced_agg['avg_exit_velocity'])
            if advanced_agg['avg_launch_angle'] is not None:
                print('True')
                avg_launch_angle.append(advanced_agg['avg_launch_angle'])
            if advanced_agg['xba'] is not None:
                xba.append(advanced_agg['xba'])
            if advanced_agg['hard_hit_rate'] is not None:
                hard_hit_rate.append(advanced_agg['hard_hit_rate'])
            if advanced_agg['chase'] is not None:
                chase.append(advanced_agg['chase'])
            if advanced_agg['whiff'] is not None:
                whiff.append(advanced_agg['whiff'])
            if advanced_agg['x'] is not None:
                x_l.append(advanced_agg['x'])
            if advanced_agg['y'] is not None:
                y_l.append(advanced_agg['y'])

    features = {
        'lineup_avg': np.mean(avg_list) if avg_list else None,
        'lineup_hr': np.mean(hr_list) if avg_list else None,

        'lineup_obp': np.mean(obp_list) if obp_list else None,
        'lineup_slg': np.mean(slg_list) if slg_list else None,
        'lineup_arhp': np.mean(vs_rhp_avg) if vs_rhp_avg else None,
        'lineup_alhp': np.mean(vs_lhp_avg) if vs_lhp_avg else None,
        'lineup_orhp': np.mean(vs_rhp_ops) if vs_rhp_ops else None,
        'lineup_olhp': np.mean(vs_lhp_ops) if vs_lhp_ops else None,
        'lineup_avg_exit_velocity': np.mean(avg_exit_velocity) if avg_exit_velocity else None,
        'lineup_launch_angle': np.mean(avg_launch_angle) if avg_launch_angle else None,
        'lineup_xba': np.mean(xba) if xba else None,
        'lineup_hard_hit_rate': np.mean(hard_hit_rate) if hard_hit_rate else None,
        'lineup_chase': np.mean(chase) if chase else None,
        'lineup_whiff': np.mean(whiff) if whiff else None,
        'x_l': np.mean(x_l) if x_l else None,
        'y_l': np.mean(y_l) if y_l else None,

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


def get_pitching_features(data, starter_id, bullpen_ids, player_ids):
    # Starter logs and rolling stats
    starter_logs = get_player_game_logs(starter_id, group='pitching')
    starter_era = compute_weighted_rolling_average(starter_logs, 'era')
    starter_whip = compute_weighted_rolling_average(starter_logs, 'whip')

    # Calculate average innings pitched
    innings_pitched = []
    for log in starter_logs:
        ip = log.get('ip')
        if ip:
            try:
                # Convert innings pitched (e.g. "5.2") to float
                parts = str(ip).split('.')
                ip_float = int(parts[0]) + int(parts[1]) / 3 if len(parts) > 1 else int(parts[0])

                innings_pitched.append(ip_float)
            except:
                continue
    avg_ip = round(np.mean(innings_pitched), 2) if innings_pitched else None

    # Get starter throwing hand
    starter_info = requests.get(f"https://statsapi.mlb.com/api/v1/people/{starter_id}").json()
    starter_hand = starter_info.get('people', [{}])[0].get('pitchHand', {}).get('code', None)  # 'R' or 'L'

    print(starter_hand)

    # Batter vs pitcher stats
    aggregated = get_bvp_stats_for_lineup(player_ids, starter_id)
    bvp_avg = aggregated['bvp_avg']
    bvp_ops = aggregated['bvp_ops']
    bvp_hr = aggregated['bvp_hr']

    advanced=get_avg_pitcher_stats(data, starter_id)

    # Bullpen ERA, WHIP
    bullpen_era_list = []
    bullpen_whip_list = []
    bull_bvp_avg=[]
    bull_bvp_ops=[]
    bull_aev=[]
    bull_ala=[]
    bull_xba=[]
    bull_hard_hit_rate=[]
    bull_whiff=[]
    bull_chase=[]
    bull_strikeout_rate=[]
    bull_walk_rate=[]
    bull_x=[]
    bull_y=[]
    right_handed_count = 0

    for pid in bullpen_ids:
        logs = get_player_game_logs(pid, group='pitching')
        if not logs:
            continue

        era = compute_weighted_rolling_average(logs, 'era')
        whip = compute_weighted_rolling_average(logs, 'whip')
        aggregated = get_bvp_stats_for_lineup(player_ids, pid)
        advanced_bull=get_avg_pitcher_stats(data, pid)


        if era is not None:
            bullpen_era_list.append(era)
        if whip is not None:
            bullpen_whip_list.append(whip)
        if aggregated is not None:
            bull_bvp_avg.append(aggregated['bvp_avg'])
        if aggregated is not None:
            bull_bvp_ops.append(aggregated['bvp_ops'])
        if advanced_bull is not None:
            bull_aev.append(advanced_bull['avg_exit_velocity'])
        if advanced_bull is not None:
            bull_ala.append(advanced_bull['avg_launch_angle'])
        if advanced_bull is not None:
            bull_xba.append(advanced_bull['xba'])

        
        if advanced_bull is not None:
            bull_hard_hit_rate.append(advanced_bull['hard_hit_rate'])
        if advanced_bull is not None:
            bull_whiff.append(advanced_bull['whiff'])
        if advanced_bull is not None:
            bull_chase.append(advanced_bull['chase'])
        if advanced_bull is not None:
            bull_strikeout_rate.append(advanced_bull['strikeout_rate'])
        if advanced_bull is not None:
            bull_walk_rate.append(advanced_bull['walk_rate'])
        if advanced_bull is not None:
            bull_x.append(advanced_bull['x'])
        if advanced_bull is not None:
            bull_y.append(advanced_bull['y'])







        # Determine throwing hand
        info = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}").json()
        hand = info.get('people', [{}])[0].get('pitchHand', {}).get('code', None)
        if hand == 'R':
            right_handed_count += 1

    bullpen_era = np.mean(bullpen_era_list) if bullpen_era_list else None
    bullpen_whip = np.mean(bullpen_whip_list) if bullpen_whip_list else None
    bullpen_bvp_avg = np.mean(bull_bvp_avg) if bull_bvp_avg else None
    bullpen_bvp_ops = np.mean(bull_bvp_ops) if bull_bvp_ops else None
    bullpen_aev = np.mean(bull_aev) if bull_aev else None
    bullpen_ala = np.mean(bull_ala) if bull_ala else None
    bullpen_xba = np.mean(bull_xba) if bull_xba else None
    bullpen_hard_hit_rate = np.mean(bull_hard_hit_rate) if bull_hard_hit_rate else None
    bullpen_whiff = np.mean(bull_whiff) if bull_whiff else None
    bullpen_chase = np.mean(bull_chase) if bull_chase else None
    bullpen_strikeout_rate = np.mean(bull_strikeout_rate) if bull_strikeout_rate else None
    bullpen_walk_rate = np.mean(bull_walk_rate) if bull_walk_rate else None
    x_b = np.mean(bull_x) if bull_x else None
    y_b = np.mean(bull_y) if bull_y else None



    bullpen_righty_pct = round(100 * right_handed_count / len(bullpen_ids), 2) if bullpen_ids else None

    return {
        'starter_era': starter_era,
        'starter_whip': starter_whip,
        'starter_avg_ip': avg_ip,
        'starter_hand': starter_hand,  # 'R', 'L', or None
        'bullpen_era': bullpen_era,
        'bullpen_whip': bullpen_whip,
        'bullpen_bvp_avg': bullpen_bvp_avg,
        'bullpen_bvp_ops': bullpen_bvp_ops,
        'bullpen_righty_pct': bullpen_righty_pct,
        'bvp_avg': bvp_avg,
        'bvp_ops': bvp_ops,
        'bvp_hr': bvp_hr, 
          'avg_exit_velocity': advanced.get('avg_exit_velocity', None) if advanced else None,
            'avg_launch_angle': advanced['avg_launch_angle'] if advanced else None,
            'xba': advanced['xba'] if advanced else None,
            'hard_hit_rate': advanced['hard_hit_rate']if advanced else None,
            'whiff': advanced['whiff'] if advanced else None,
            'chase': advanced['chase'] if advanced else None,
            'strikeout_rate':advanced['strikeout_rate'] if advanced else None,
            'x_s': advanced['x'] if advanced else None,
            'y_s': advanced['y'] if advanced else None,
            'walk_rate': advanced['walk_rate'] if advanced else None,
            'bull_avg_exit_velocity': bullpen_aev,
            'bull_avg_launch_angle': bullpen_ala,
            'bull_xba': bullpen_xba,
            'bull_hard_hit_rate': bullpen_hard_hit_rate,
            'bull_whiff': bullpen_whiff,
            'bull_chase': bullpen_chase,
            'bull_strikeout_rate':bullpen_strikeout_rate,
            'bull_walk_rate': bullpen_walk_rate,
            'x_b':x_b,
            'y_b': y_b


    }





def get_avg_batter_stats(data, batter_id):
    try:
        batter_data = data[data['batter'] == batter_id]
        swing_data=swing_stats(batter_data)
        batted_balls = batter_data[batter_data['launch_speed'].notnull()]

        if batted_balls.empty:
            return None

        avg_stats = {
            'avg_exit_velocity': round(batted_balls['launch_speed'].mean(), 2),
            'avg_launch_angle': round(batted_balls['launch_angle'].mean(), 2) if 'launch_angle' in batted_balls else None,
            'xba': round(batted_balls['estimated_ba_using_speedangle'].mean(), 3) if 'estimated_ba_using_speedangle' in batted_balls else None,
            'hard_hit_rate': round((batted_balls['launch_speed'] >= 95).sum() / len(batted_balls), 3),
            'whiff': swing_data['whiffpct'],
            'chase': swing_data['chasepct'],
            'x': round(batted_balls['hc_x'].mean(), 2),
            'y': round(batted_balls['hc_y'].mean(), 2)


        }

        return avg_stats

    except Exception as e:
        print(f"Error fetching data for batter {batter_id}: {e}")
        return None
    
def swing_stats(data):
    swing_balls = data[data['description'].isin({'foul', 'swinging_strike', 'hit_into_play', 'swinging_strike_blocked'})]
    total_swings=len(swing_balls)

    whiff=swing_balls[swing_balls['description'].isin( {'swinging_strike', 'swinging_strike_blocked'})]

    chase=swing_balls[swing_balls['zone'].isin({11,12,13,14})]


    return {'whiffpct': len(whiff)/total_swings, 'chasepct': len(chase)/total_swings}
    



def get_avg_pitcher_stats(data, player_id):
    try:
        batter_data = data[data['pitcher'] == player_id]
        swing_data=swing_stats(batter_data)

        batted_balls = batter_data[batter_data['launch_speed'].notnull()]

        if batted_balls.empty:
            return None

        strikeouts = batter_data[batter_data['events'] == 'strikeout']
        walks=batter_data[batter_data['events'] == 'walk']
        strikeout_rate = len(strikeouts) / batter_data['at_bat_number'].nunique()
        walk_rate= len(walks) / batter_data['at_bat_number'].nunique()



        avg_stats = {
            'avg_exit_velocity': round(batted_balls['launch_speed'].mean(), 2),
            'avg_launch_angle': round(batted_balls['launch_angle'].mean(), 2) if 'launch_angle' in batted_balls else None,
            'xba': round(batted_balls['estimated_ba_using_speedangle'].mean(), 3) if 'estimated_ba_using_speedangle' in batted_balls else None,
            'hard_hit_rate': round((batted_balls['launch_speed'] >= 95).sum() / len(batted_balls), 3),
            'whiff': swing_data['whiffpct'],
            'chase': swing_data['chasepct'],
            'strikeout_rate':strikeout_rate,
            'walk_rate': walk_rate,
            'x': round(batted_balls['hc_x'].mean(), 2),
            'y': round(batted_balls['hc_y'].mean(), 2)


        }

        return avg_stats

    except Exception as e:
        print(f"Error fetching data for pitcher {player_id}: {e}")
        return None
    
games = get_schedule(START_DATE, END_DATE)
data_rows = []


i=0

for game in games:
    print(i)
    i=i+1
    gamePk = game['gamePk']
    stadium=game['venue']
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
        team_features = get_team_features(data, team_id, lineup_player_ids)
        stadium_features=get_stadium_features(stadium)
        pitching_features = get_pitching_features(data, starter_id, bullpen_ids, lineup_player_ids)

        hits = team_info['teamStats']['batting'].get('hits')
        runs=team_info['teamStats']['batting'].get('runs')

        if hits is None:
            continue
        if runs is None:
            print("Runs false")
            continue

        row = {
            'team_id': team_id,
            'is_home': team_type == 'home',
            'hits': hits,
            'runs': runs
        }
        row.update(team_features)
        row.update(pitching_features)
        row.update(stadium_features)
        data_rows.append(row)




df = pd.DataFrame(data_rows)
df.to_csv("processed_hits_dataset.csv", index=False)



    


    
