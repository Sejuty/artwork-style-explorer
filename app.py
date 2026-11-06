"""Streamlit app: interactive painting map + 'find similar paintings'.

Needs data/metadata.csv, data/embeddings.npy (from the scripts) and
data/tsne.npy, data/clusters.npy (saved at the end of the notebook).

    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from prepare_data import DATA, IMAGES, META

st.set_page_config(page_title="Artwork Style Explorer", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv(META)
    emb = np.load(DATA / "embeddings.npy")
    xy = np.load(DATA / "tsne.npy")
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df["cluster"] = np.load(DATA / "clusters.npy").astype(str)
    return df, emb


try:
    df, emb = load()
except FileNotFoundError as e:
    st.error(f"Missing {e.filename}. Run prepare_data.py, embed.py, then the notebook first.")
    st.stop()

st.title("Artwork Style Explorer")
map_tab, similar_tab = st.tabs(["Painting map", "Find similar"])

with map_tab:
    color = st.radio("Color by", ["cluster", "style", "genre"], horizontal=True)
    fig = px.scatter(df, x="x", y="y", color=color, hover_data=["style", "artist", "genre"],
                     height=700, opacity=0.7)
    fig.update_traces(marker_size=4)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with similar_tab:
    idx = st.number_input("Painting index", 0, len(df) - 1, 0)
    if st.button("Random painting"):
        idx = int(np.random.randint(len(df)))
    row = df.iloc[idx]
    st.image(str(IMAGES / row.file), caption=f"{row.artist} · {row['style']}", width=300)

    # Embeddings are unit length, so a dot product is the cosine similarity.
    sims = emb @ emb[idx]
    top = np.argsort(-sims)[1:7]  # [0] is the painting itself
    cols = st.columns(6)
    for col, j in zip(cols, top):
        r = df.iloc[j]
        col.image(str(IMAGES / r.file), caption=f"{r['style']} ({sims[j]:.2f})")
