"""PCA: how many of CLIP's 512 dimensions actually matter? (course ch. 3)

Keeps the fewest components that explain --variance of the total. Clustering
on fewer dimensions is faster and less noisy.

    python src/reduce.py                 # keep 90% of the variance
    python src/reduce.py --variance 0.95
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import SEED, load_embeddings, require
from paths import EMBEDDINGS, FIGURES_DIR, PCA_EMBEDDINGS, PCA_JSON, ensure_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variance", type=float, default=0.90, help="share of variance to keep")
    args = p.parse_args()

    require(EMBEDDINGS)
    ensure_dirs()
    _, X = load_embeddings()

    cumulative = PCA(random_state=SEED).fit(X).explained_variance_ratio_.cumsum()
    n = int(np.searchsorted(cumulative, args.variance) + 1)

    X_pca = PCA(n_components=n, random_state=SEED).fit_transform(X).astype(np.float32)
    np.save(PCA_EMBEDDINGS, X_pca)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    ax.axhline(args.variance, color="grey", ls="--", lw=1)
    ax.axvline(n, color="grey", ls="--", lw=1)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"{n} of {X.shape[1]} components keep {args.variance:.0%} of the variance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)

    with open(PCA_JSON, "w") as fh:
        json.dump({
            "input_dims": int(X.shape[1]),
            "target_variance": args.variance,
            "n_components": n,
            "variance_kept": round(float(cumulative[n - 1]), 4),
            "components_for": {f"{v:.0%}": int(np.searchsorted(cumulative, v) + 1)
                               for v in (0.5, 0.8, 0.9, 0.95, 0.99)},
        }, fh, indent=2)

    print(f"{n} components keep {cumulative[n - 1]:.1%} of the variance -> {PCA_EMBEDDINGS}")


if __name__ == "__main__":
    main()
