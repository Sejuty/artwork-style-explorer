# What each file does

All scripts live in `src/` and run from anywhere: `python src/<name>.py`.

| File | Reads | Writes | Purpose |
|---|---|---|---|
| `paths.py` | — | — | Every file location, resolved from the repo root. |
| `common.py` | — | — | Shared loaders (`load_metadata`, `load_embeddings`), the `require` check, the thumbnail-grid plot and the random seed. |
| `prepare_data.py` | Hugging Face `huggan/wikiart` (streamed) | `data/images/*.jpg`, `data/metadata.csv` | Keeps up to `--per-style` paintings per style as thumbnails. Each metadata row records its position in the shuffled stream (`row`), so a restart skips what was already read. |
| `embed.py` | `data/images`, `data/metadata.csv` | `data/embeddings.npy` | CLIP ViT-B/32 (`laion2b_s34b_b79k`) embedding per painting, unit length. Uses CUDA, then Apple MPS, then CPU. Also exposes `load_model`, `embed_images` and `embed_texts` for `predict.py` and the app. |
| `show_samples.py` | `data/metadata.csv` | `docs/figures/paintings_random.png` | 4×4 grid of random paintings with their style. |
| `reduce.py` | `data/embeddings.npy` | `data/pca.npy`, `results/pca.json`, `pca_variance.png` | PCA keeping `--variance` (default 90%). |
| `cluster.py` | `data/pca.npy`, `data/metadata.csv` | `data/clusters.npy`, `results/clusters.json`, `elbow.png`, `crosstab.png`, `paintings_purest_clusters.png`, `paintings_mixed_clusters.png` | Elbow over k = 5…40, k-means at `--k` (default 27), ARI/NMI and per-cluster/per-style breakdowns against the real styles. |
| `hierarchy.py` | `data/embeddings.npy`, `data/metadata.csv` | `dendrogram.png` | Average-linkage, cosine-distance dendrogram of the per-style mean embeddings. |
| `tsne_map.py` | `data/pca.npy`, `data/clusters.npy` | `data/tsne.npy`, `tsne.png` | 2D t-SNE map, coloured by cluster and by style. |
| `pixel_baseline.py` | `data/images`, `results/pca.json`, `results/clusters.json` | `results/pixel_baseline.json` | Same PCA → k-means → ARI on 32×32 grayscale pixels, for comparison with CLIP. |
| `predict.py` | `data/embeddings.npy`, `data/metadata.csv` | `results/upload_accuracy.json` (with `--evaluate`) | Guesses genre and style for a new picture: weighted vote of its 15 nearest paintings, plus a CLIP "is this a painting?" check. `--evaluate` measures accuracy leave-one-out on the collection. |
| `app.py` | `data/*.npy`, `data/metadata.csv`, `results/*.json` | — | Streamlit app: t-SNE painting map (click a dot to open it), upload a picture, similar-painting search, cluster browser, and an explanation page. |

Figures go to `docs/figures/`. The `paintings_*.png` grids contain the
copyrighted thumbnails and are git-ignored; the charts are committed.
