"""Loaders and plotting helpers shared by the analysis scripts.

The rule for the whole analysis: `style` is never used to fit anything. It only
comes out at the end, to grade the clusters (and to describe styles in
hierarchy.py, which fits nothing).
"""

import matplotlib
matplotlib.use("Agg")  # scripts only save figures, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from paths import EMBEDDINGS, IMAGES_DIR, METADATA_CSV

SEED = 42


def load_metadata():
    return pd.read_csv(METADATA_CSV)


def load_embeddings():
    """Metadata and CLIP embeddings, checked to line up row for row."""
    df = load_metadata()
    X = np.load(EMBEDDINGS)
    if len(df) != len(X):
        raise SystemExit(f"{METADATA_CSV.name} has {len(df)} rows but {EMBEDDINGS.name} "
                         f"has {len(X)}. Re-run embed.py.")
    return df, X


def require(*paths):
    """Stop with a clear message if an earlier step hasn't been run."""
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run the earlier steps first (see README).")


def thumbnail_grid(df, rows, cols, titles, path, suptitle=None, cell=2.0):
    """Save a rows x cols grid of the paintings in df (in order) with titles."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell + 0.4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, row), title in zip(np.ravel(axes), df.iterrows(), titles):
        ax.imshow(Image.open(IMAGES_DIR / row["file"]))
        ax.set_title(title, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
