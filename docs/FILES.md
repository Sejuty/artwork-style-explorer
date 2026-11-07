# What each file does

All scripts live in `src/` and run from anywhere: `python src/<name>.py`.

| File | Reads | Writes | Purpose |
|---|---|---|---|
| `paths.py` | — | — | Every file location, resolved from the repo root. |
| `common.py` | — | — | Shared loaders (`load_metadata`, `load_embeddings`), the `require` check, the thumbnail-grid plot and the random seed. |
| `prepare_data.py` | Hugging Face `huggan/wikiart` (streamed) | `data/images/*.jpg`, `data/metadata.csv` | Keeps up to `--per-style` paintings per style as thumbnails. Each metadata row records its position in the shuffled stream (`row`), so a restart skips what was already read. |
| `embed.py` | `data/images`, `data/metadata.csv` | `data/embeddings.npy` | CLIP ViT-B/32 (`laion2b_s34b_b79k`) embedding per painting, unit length. Uses CUDA, then Apple MPS, then CPU. |
| `show_samples.py` | `data/metadata.csv` | `docs/figures/paintings_random.png` | 4×4 grid of random paintings with their style. |
| `reduce.py` | `data/embeddings.npy` | `data/pca.npy`, `results/pca.json`, `pca_variance.png` | PCA keeping `--variance` (default 90%). |
| `cluster.py` | `data/pca.npy`, `data/metadata.csv` | `data/clusters.npy`, `results/clusters.json`, `elbow.png`, `crosstab.png`, `paintings_purest_clusters.png`, `paintings_mixed_clusters.png` | Elbow over k = 5…40, k-means at `--k` (default 27), ARI/NMI and per-cluster/per-style breakdowns against the real styles. |
| `hierarchy.py` | `data/embeddings.npy`, `data/metadata.csv` | `dendrogram.png` | Average-linkage, cosine-distance dendrogram of the per-style mean embeddings. |
| `tsne_map.py` | `data/pca.npy`, `data/clusters.npy` | `data/tsne.npy`, `tsne.png` | 2D t-SNE map, coloured by cluster and by style. |
| `pixel_baseline.py` | `data/images`, `results/pca.json`, `results/clusters.json` | `results/pixel_baseline.json` | Same PCA → k-means → ARI on 32×32 grayscale pixels, for comparison with CLIP. |
| `app.py` | `data/*.npy`, `data/metadata.csv` | — | Streamlit app: t-SNE painting map (colour by cluster / style / genre) and cosine-similarity search. |

Figures go to `docs/figures/`. The `paintings_*.png` grids contain the
copyrighted thumbnails and are git-ignored; the charts are committed.
