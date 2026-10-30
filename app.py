import streamlit as st
import pandas as pd
from data import load_players
from similarity import build_vectors, find_comps

st.title("NBAComps")
st.write("Find NBA players with similar playing styles, based on 2025-26 per-36 stats.")

@st.cache_data
def get_data():
    df = load_players()
    return df, build_vectors(df)

df, X_unit = get_data()

name = st.selectbox("Choose a player", sorted(df["PLAYER_NAME"]))
k = st.slider("Number of comps", 3, 10, 5)

comps = find_comps(df, X_unit, name, k)
st.subheader(f"Players most similar to {name}")
st.table(pd.DataFrame(comps, columns=["Player", "Similarity"]))