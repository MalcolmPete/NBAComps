import unicodedata
import torch
from data import FEATURES

def simplify(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()

def build_vectors(df):
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    X = (X - X.mean(0)) / X.std(0)
    return X / X.norm(dim=1, keepdim=True)

def find_comps(df, X_unit, name, k=5):
    matches = df.index[df["PLAYER_NAME"].apply(simplify) == simplify(name)]
    if len(matches) == 0:
        return None
    i = matches[0]
    sims = X_unit @ X_unit[i]
    top = sims.argsort(descending=True)[1:k + 1]
    return [(df.loc[j.item(), "PLAYER_NAME"], round(sims[j].item(), 3)) for j in top]