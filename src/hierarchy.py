"""Hierarchical clustering of the styles: which movements are related?
(course ch. 2)

Averages the CLIP embeddings per style and builds a dendrogram. Using the
labels is fine here: this describes styles, it doesn't fit a model to them.

    python src/hierarchy.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from common import load_embeddings, require
from paths import EMBEDDINGS, FIGURES_DIR, ensure_dirs


def main():
    require(EMBEDDINGS)
    ensure_dirs()
    df, X = load_embeddings()

    means = pd.DataFrame(X).groupby(df["style"]).mean()
    Z = linkage(means.values, method="average", metric="cosine")

    fig, ax = plt.subplots(figsize=(8, 9))
    dendrogram(Z, labels=means.index.str.replace("_", " ").tolist(),
               orientation="left", ax=ax, leaf_font_size=9)
    ax.set_xlabel("cosine distance between style averages")
    ax.set_title("Which art movements look alike?")
    fig.tight_layout()
    out = FIGURES_DIR / "dendrogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
