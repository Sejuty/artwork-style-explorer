"""Why embeddings? The same PCA -> k-means -> ARI pipeline on raw pixels.

Each painting becomes a 32x32 grayscale image, flattened to 1024 numbers. If
CLIP is worth using, this should score far below cluster.py.

    python src/pixel_baseline.py
"""
import json

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import SEED, load_metadata, require
from paths import (CLUSTERS_JSON, IMAGES_DIR, PCA_JSON, PIXEL_BASELINE_JSON,
                   ensure_dirs)

SIZE = 32


def main():
    require(CLUSTERS_JSON, PCA_JSON)
    ensure_dirs()
    df = load_metadata()
    clip = json.loads(CLUSTERS_JSON.read_text())
    variance = json.loads(PCA_JSON.read_text())["target_variance"]
    k = clip["k"]

    print(f"Loading {len(df):,} images as {SIZE}x{SIZE} grayscale ...")
    X = np.stack([np.asarray(Image.open(IMAGES_DIR / f).convert("L").resize((SIZE, SIZE)),
                             dtype=np.float32).ravel() / 255.0 for f in df["file"]])

    pca = PCA(n_components=variance, svd_solver="full", random_state=SEED)
    X_pca = pca.fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_pca)
    ari = adjusted_rand_score(df["style"], labels)
    nmi = normalized_mutual_info_score(df["style"], labels)

    result = {
        "pixels": {"dims": int(X.shape[1]), "pca_components": int(pca.n_components_),
                   "ari": round(ari, 4), "nmi": round(nmi, 4)},
        "clip": {"ari": clip["ari"], "nmi": clip["nmi"]},
        "k": k,
    }
    PIXEL_BASELINE_JSON.write_text(json.dumps(result, indent=2))
    print(f"k={k}   raw pixels: ARI {ari:.3f}  NMI {nmi:.3f}")
    print(f"        CLIP:       ARI {clip['ari']:.3f}  NMI {clip['nmi']:.3f}")
    print(f"Saved {PIXEL_BASELINE_JSON}")


if __name__ == "__main__":
    main()
