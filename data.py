import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

FEATURES = ["PTS", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "FG3_PCT", "FT_PCT"]

def load_players(season="2025-26", min_minutes=500):
    cache = f"players_{season}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="Per36"
        ).get_data_frames()[0]
        df.to_csv(cache, index=False)

    return df[df["MIN"] >= min_minutes].reset_index(drop=True)