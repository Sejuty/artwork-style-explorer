"""t-SNE: a 2D map of every painting (course ch. 2). The app plots these
coordinates.

    python src/tsne_map.py        # a few minutes
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from common import SEED, load_metadata, require
from paths import CLUSTERS, FIGURES_DIR, PCA_EMBEDDINGS, TSNE as TSNE_PATH, ensure_dirs

PERPLEXITY = 30


def scatter(ax, xy, labels, title):
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab20", 20)
    markers = ["o", "^", "s"]  # 20 colours x 3 markers covers up to 60 groups
    for i, c in enumerate(cats):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.6, color=cmap(i % 20),
                   marker=markers[i // 20 % 3], label=str(c).replace("_", " "))
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    require(PCA_EMBEDDINGS, CLUSTERS)
    ensure_dirs()
    df = load_metadata()
    X = np.load(PCA_EMBEDDINGS)
    clusters = np.load(CLUSTERS)

    print(f"t-SNE on {X.shape} ...")
    xy = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca",
              random_state=SEED).fit_transform(X).astype(np.float32)
    np.save(TSNE_PATH, xy)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    scatter(axes[0], xy, clusters, "Coloured by k-means cluster")
    scatter(axes[1], xy, df["style"].to_numpy(), "Coloured by real style")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8,
                   markerscale=4, frameon=False)
    fig.tight_layout()
    out = FIGURES_DIR / "tsne.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {TSNE_PATH} and {out}")


if __name__ == "__main__":
    main()
