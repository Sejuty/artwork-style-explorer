"""Every filesystem location the project uses, resolved from this file rather
than from the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"          # rebuilt on every machine, never committed
RESULTS_DIR = ROOT / "results"    # small JSON summaries, committed
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# Step 1: prepare_data.py
IMAGES_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"

# Step 2: embed.py
EMBEDDINGS = DATA_DIR / "embeddings.npy"      # (N, 512) CLIP, unit length

# Steps 3-5: reduce.py, cluster.py, tsne_map.py
PCA_EMBEDDINGS = DATA_DIR / "pca.npy"         # (N, n_components)
CLUSTERS = DATA_DIR / "clusters.npy"          # (N,) k-means label per painting
TSNE = DATA_DIR / "tsne.npy"                  # (N, 2) map coordinates for the app

PCA_JSON = RESULTS_DIR / "pca.json"
CLUSTERS_JSON = RESULTS_DIR / "clusters.json"
PIXEL_BASELINE_JSON = RESULTS_DIR / "pixel_baseline.json"


def ensure_dirs():
    """Create the output directories the scripts write into."""
    for directory in (DATA_DIR, IMAGES_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
