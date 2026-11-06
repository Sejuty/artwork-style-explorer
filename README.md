# Artwork Style Explorer

Can a model rediscover art movements without being told what they are?
This project clusters ~10k WikiArt paintings by how they look (CLIP embeddings → PCA → k-means), grades the clusters against the real style labels, and ships a small app with an interactive painting map and a "find similar paintings" search.

Built on the techniques from DataCamp's *Unsupervised Learning in Python*: k-means, hierarchical clustering, t-SNE, PCA, and cosine-similarity recommenders.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
# GPU (NVIDIA)? Install torch first with the command from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Run order
| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python prepare_data.py` | Streams the 34 GB dataset from Hugging Face and keeps up to 400 paintings per style as 224px thumbnails in `data/` (~10k images, ~200 MB). Safe to stop and restart: it resumes. | 1–2 h, depends on internet |
| 2 | `python embed.py` | CLIP turns each painting into 512 numbers → `data/embeddings.npy` | ~5 min GPU / 20–40 min CPU |
| 3 | `jupyter notebook notebooks/01_explore.ipynb` | **Your part:** PCA, k-means, crosstab, dendrogram, t-SNE. The last cell saves results for the app. | — |
| 4 | `streamlit run app.py` | Painting map + similar-painting search | — |

Quick test before the full download: `python prepare_data.py --per-style 5 --max-rows 2000`. Delete `data/` afterwards.

`data/` is **not in git** on purpose. Every machine rebuilds it with step 1, and the paintings are copyrighted.

## Dataset
[`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart): 81,444 paintings, 27 styles, 11 genres, 129 artists.
- The 72 shard files are **not shuffled** (the early ones are missing many styles), so `prepare_data.py` shuffles shard order and caps each style.
- Styles are very unbalanced (Impressionism ≈ 13k paintings, some styles only a few hundred). The cap keeps the clusters from all becoming Impressionism.
- License: unknown, © original artists. Use it for learning and portfolio work only, and don't redistribute the images.

## Stretch goals
- NMF on the embeddings: interpretable "visual parts" (course ch. 4)
- Text search with CLIP: type "stormy sea at night" and get matching paintings
- Deploy the app (Streamlit Community Cloud / Hugging Face Spaces)
