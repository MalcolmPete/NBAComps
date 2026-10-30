from data import load_players
from similarity import build_vectors, find_comps

df = load_players()
X_unit = build_vectors(df)

for name in ["Stephen Curry", "Nikola Jokic"]:
    comps = find_comps(df, X_unit, name)
    print(f"\nMost similar to {name}:")
    if comps is None:
        print("  not found")
    else:
        for player, score in comps:
            print(f"  {player:<25} {score}")