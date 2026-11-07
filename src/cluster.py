"""k-means on the PCA embeddings, then grade the clusters against the real
styles. (course ch. 1: k-means, the elbow, the fish crosstab)

Style labels are not used to fit anything here; they only grade the result.

    python src/cluster.py           # k = 27, the number of real styles
    python src/cluster.py --k 20
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import SEED, load_metadata, require, thumbnail_grid
from paths import (CLUSTERS, CLUSTERS_JSON, FIGURES_DIR, PCA_EMBEDDINGS,
                   ensure_dirs)

ELBOW_KS = range(5, 41, 5)
N_INIT = 10
GRID_CLUSTERS = 6    # clusters shown in each paintings grid
GRID_PER_CLUSTER = 8


def kmeans(X, k):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)


def elbow(X, styles):
    """Inertia (what k-means minimises) and ARI (graded against styles) per k."""
    rows = []
    for k in ELBOW_KS:
        km = kmeans(X, k)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "ari": float(adjusted_rand_score(styles, km.labels_))})
        print(f"  k={k:2d}  inertia={km.inertia_:9.1f}  ARI={rows[-1]['ari']:.3f}")
    return rows


def plot_elbow(rows, k, path):
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, [r["inertia"] for r in rows], "o-", label="inertia")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("inertia (lower = tighter clusters)")
    ax2 = ax.twinx()
    ax2.plot(ks, [r["ari"] for r in rows], "s--", color="tab:orange", label="ARI vs styles")
    ax2.set_ylabel("adjusted Rand index vs real styles", color="tab:orange")
    ax.axvline(k, color="grey", ls=":", lw=1)
    ax.set_title(f"Elbow plot (chosen k = {k})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crosstab(share, path):
    """Rows = styles, columns = clusters, cell = share of the style in that cluster.
    Clusters are ordered by their dominant style so a clean result is a diagonal."""
    order = share.idxmax(axis=0).map({s: i for i, s in enumerate(share.index)})
    share = share[order.sort_values(kind="stable").index]
    fig, ax = plt.subplots(figsize=(0.32 * share.shape[1] + 4, 0.32 * share.shape[0] + 1.5))
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(share.index)), share.index.str.replace("_", " "), fontsize=8)
    ax.set_xticks(range(share.shape[1]), share.columns, fontsize=7)
    ax.set_xlabel("cluster")
    ax.set_title("Where each style's paintings end up (share of the style per cluster)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cluster_grid(df, clusters, ids, summary, path, title):
    picks, titles = [], []
    for c in ids:
        members = df[clusters == c].sample(GRID_PER_CLUSTER, random_state=SEED, replace=False)
        picks.append(members)
        top = summary[c]["top_styles"][0]
        titles += [f"c{c}: {top['style'].replace('_', ' ')} {top['share']:.0%}\n"
                   f"{s.replace('_', ' ')}" for s in members["style"]]
    thumbnail_grid(pd.concat(picks), len(ids), GRID_PER_CLUSTER, titles, path,
                   suptitle=title, cell=1.7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=27, help="number of clusters (27 = number of real styles)")
    args = p.parse_args()

    require(PCA_EMBEDDINGS)
    ensure_dirs()
    df = load_metadata()
    X = np.load(PCA_EMBEDDINGS)
    styles = df["style"]

    print("Elbow:")
    rows = elbow(X, styles)

    km = kmeans(X, args.k)
    clusters = km.labels_
    np.save(CLUSTERS, clusters)

    ari = adjusted_rand_score(styles, clusters)
    nmi = normalized_mutual_info_score(styles, clusters)
    counts = pd.crosstab(styles, clusters)
    share = counts.div(counts.sum(axis=1), axis=0)

    summary = {}
    for c in counts.columns:
        col = counts[c].sort_values(ascending=False)
        summary[int(c)] = {
            "size": int(col.sum()),
            "top_styles": [{"style": s, "share": round(float(n / col.sum()), 3)}
                           for s, n in col.head(3).items() if n],
        }
    purity = {c: s["top_styles"][0]["share"] for c, s in summary.items()}

    per_style = {}
    for s in share.index:
        best = int(share.loc[s].idxmax())
        per_style[s] = {"paintings": int(counts.loc[s].sum()), "best_cluster": best,
                        "share_in_best_cluster": round(float(share.loc[s, best]), 3)}
    per_style = dict(sorted(per_style.items(), key=lambda kv: -kv[1]["share_in_best_cluster"]))

    plot_elbow(rows, args.k, FIGURES_DIR / "elbow.png")
    plot_crosstab(share, FIGURES_DIR / "crosstab.png")
    by_purity = sorted(purity, key=purity.get, reverse=True)
    cluster_grid(df, clusters, by_purity[:GRID_CLUSTERS], summary,
                 FIGURES_DIR / "paintings_purest_clusters.png", "Purest clusters")
    cluster_grid(df, clusters, by_purity[-GRID_CLUSTERS:], summary,
                 FIGURES_DIR / "paintings_mixed_clusters.png", "Most mixed clusters")

    with open(CLUSTERS_JSON, "w") as fh:
        json.dump({"k": args.k, "n_paintings": len(df), "ari": round(ari, 4),
                   "nmi": round(nmi, 4), "elbow": rows, "per_style": per_style,
                   "clusters": summary}, fh, indent=2)

    print(f"\nk={args.k}: ARI = {ari:.3f}, NMI = {nmi:.3f}  (0 = random, 1 = perfect)")
    print("\nStyles found most cleanly (share of the style in its main cluster):")
    for s, v in list(per_style.items())[:6]:
        print(f"  {s:30s} {v['share_in_best_cluster']:.0%}")
    print("Most scattered:")
    for s, v in list(per_style.items())[-6:]:
        print(f"  {s:30s} {v['share_in_best_cluster']:.0%}")
    print(f"\nSaved {CLUSTERS}, {CLUSTERS_JSON} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
