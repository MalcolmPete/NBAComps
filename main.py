from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import torch
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)

import unicodedata

def simplify(name):
    # "Nikola Jokić" -> "nikola jokic"
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()

df = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26", per_mode_detailed="Per36"
).get_data_frames()[0]

print(df.shape)
print(df.columns.tolist())
print(df[["PLAYER_NAME", "GP", "MIN", "PTS", "REB", "AST", "FG3A", "STL", "BLK"]].head(10))

# Features
df = df[df["MIN"] >= 500].reset_index(drop=True)   # enough minutes to trust the per-36 numbers

FEATURES = ["PTS", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "FG3_PCT", "FT_PCT"]

X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
X = (X - X.mean(0)) / X.std(0)

print(df.shape)
print(X.shape)
print(df["PLAYER_NAME"][:3].tolist())
print(X[:3])

# Cosine similarity
X_unit = X / X.norm(dim=1, keepdim=True)   # scale each player's vector to length 1

def find_comps(name, k=5):
    matches = df.index[df["PLAYER_NAME"].apply(simplify) == simplify(name)]
    if len(matches) == 0:
        print(f"{name} not found")
        return
    i = matches[0]

    sims = X_unit @ X_unit[i]               # similarity to every player
    top = sims.argsort(descending=True)[1:k + 1]   # skip [0], it's the player himself

    print(f"\nMost similar to {df.loc[i, 'PLAYER_NAME']}:")
    for j in top:
        j = j.item()
        print(f"  {df.loc[j, 'PLAYER_NAME']:<25} {sims[j].item():.3f}")

find_comps("Stephen Curry")
find_comps("Nikola Jokic")