# NBAComps

Finds NBA players with similar playing styles using per-36 stats.

**Live app:** https://nbacomps.streamlit.app/

Each player is described by 12 per-36 stats (scoring, shot profile, rebounding, playmaking, defense and shooting), normalized and compared with cosine similarity.

Still in development. Next up is a PyTorch autoencoder to learn player embeddings and compare them against the raw-stat comps.

Built with Python, pandas, PyTorch, Streamlit and nba_api.

## Run

    pip install -r requirements.txt
    streamlit run app.py