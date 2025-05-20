import requests

API_KEY = '02b7acbdd6c90012b7a73cfc032d670542e853de2a503f804a4fc69e8d15049e'
url = 'https://api.odds-api.io/v2/events'
url2 = 'https://api.odds-api.io/v2/odds'

params = {
    'apiKey': API_KEY,
    'sport': 'baseball',
    'league': 'mlb',
    'status': 'pending',
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Print the data to understand structure
    print("Events data:", data)

    # Depending on the API, 'data' might be a dict with a key like 'events' or just a list
    events = data.get('events') if isinstance(data, dict) else data

    for game in events:
        event_id = game.get('id')
        print(f"Processing event ID: {event_id}")

        newParams = {
            'apiKey': API_KEY,
            'eventId': event_id,
            'bookmakers': 'Bet365',
        }

        response2 = requests.get(url2, params=newParams)
        response2.raise_for_status()
        data2 = response2.json()
        print("Odds data for event:", data2)

        # Check where bookmakers exist in the response
        # Possibly data2 has a key 'data' or 'odds', check with a print first
        odds_data = data2.get('data') or data2.get('odds') or data2

        # You may need to iterate over odds_data to find bookmakers
        # Here I assume odds_data is a dict with 'bookmakers' key
        bookmakers = odds_data.get('bookmakers') if isinstance(odds_data, dict) else None

        if bookmakers and 'Bet365' in bookmakers:
            bet365_odds = bookmakers['Bet365']

            # bet365_odds is a list — find the totals entry
            totals_entry = next((item for item in bet365_odds if item.get('name') == 'Totals'), None)

            if totals_entry:
                totals_odds = totals_entry.get('odds')

                if isinstance(totals_odds, list):
                    for odd in totals_odds:
                        hdp = odd.get('hdp')  # Handicap (e.g., Over/Under line)
                        over = odd.get('over')
                        under = odd.get('under')
                        print(f"HDP: {hdp}, Over: {over}, Under: {under}")
                else:
                    print("Unexpected format for totals_odds:", totals_odds)
        else:
            print("Bet365 bookmakers not found in odds data")

except requests.exceptions.RequestException as e:
    print("Error calling the Odds API:", e)
except Exception as e:
    print("Unexpected error:", e)
