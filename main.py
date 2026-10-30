from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import torch
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)

df = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26", per_mode_detailed="Per36"
).get_data_frames()[0]

print(df.shape)
print(df.columns.tolist())
print(df[["PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST", "FG3A", "STL", "BLK"]].head(10))

# Step 2: Features
df = df[df["MIN"] >= 500].reset_index(drop=True)   # enough minutes to trust the per-36 numbers

FEATURES = ["PTS", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "FG3_PCT", "FT_PCT"]

X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
X = (X - X.mean(0)) / X.std(0)

print(df.shape)
print(X.shape)
print(df["PLAYER_NAME"][:3].tolist())
print(X[:3])