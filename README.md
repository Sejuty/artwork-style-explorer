# Artwork Style Explorer

Can a model rediscover art movements without being told what they are?
This project clusters ~10k WikiArt paintings by how they look (CLIP embeddings → PCA → k-means), grades the clusters against the real style labels, and ships a Streamlit app to explore the result: an interactive style map, a page that guesses the subject and style of an uploaded picture, a cluster browser, and a plain-language walk through every chart.

Built on the techniques from DataCamp's *Unsupervised Learning in Python*: k-means, hierarchical clustering, t-SNE, PCA, and cosine-similarity recommenders.

**Result.** With k = 27 (one cluster per real style), k-means on CLIP embeddings
scores ARI 0.189 / NMI 0.398 against the real styles; the same pipeline on raw
pixels scores ARI 0.019, ten times lower. Ukiyo-e is found almost perfectly
(one cluster, 99% pure), and the three Cubisms share one cluster. The
neighbouring movements — Impressionism / Post-Impressionism, the four
Renaissance styles, Expressionism / Fauvism — are spread over several mixed
clusters. A dendrogram of the style averages rebuilds art history without being
told any dates: Renaissance styles together, Baroque with Rococo, the Cubisms
together, and the abstract movements on their own branch.

The mixed clusters are not random: most group paintings by **subject or
medium** (portraits, nudes, pencil and ink drawings, buildings and streets)
rather than by movement. The model finds real structure; it just isn't always
the structure art historians use, which is the main reason the ARI isn't higher.

The app's **upload** page guesses the subject and style of a new picture from
its 15 closest paintings. Tested leave-one-out on the collection, the first
guess is right 81% of the time for subject and 58% for style (96% / 83% within
the top three).

Per-file reference: [`docs/FILES.md`](docs/FILES.md).

## Layout

```
  huggan/wikiart (Hugging Face, 81k paintings, streamed)
          |
          v
  prepare_data.py        400 per style, 224px thumbnails
          |
          v
  data/images/  data/metadata.csv   (~10k paintings)
          |
          +------------------------------+
          v                              v
  embed.py   CLIP ViT-B/32         show_samples.py   random grid
          |                        pixel_baseline.py raw-pixel comparison
          v
  data/embeddings.npy  (N x 512)
          |
          +------------------------------+
          v                              v
  reduce.py   PCA, keep 90%          hierarchy.py   dendrogram of styles
          |
          v
  data/pca.npy
          |
          v
  cluster.py   elbow, k-means, grade vs styles  -> results/clusters.json
          |
          v
  data/clusters.npy
          |
          v
  tsne_map.py  2D map                -> data/tsne.npy
          |
          v
  app.py   style map, upload, clusters, chart explanations
             ^
             |
  predict.py   k-nearest-neighbour guesses for an uploaded picture
               (--evaluate -> results/upload_accuracy.json)
```

```
.
├── src/            all Python scripts — see docs/FILES.md
├── data/           images, metadata, embeddings (not in git, rebuilt by step 1–2)
├── results/        small JSON summaries of each analysis step
├── docs/           FILES.md, figures/
└── requirements.txt
```

Scripts resolve their paths from the repository root (`src/paths.py`) rather
than the working directory, so `python src/cluster.py` and
`cd src && python cluster.py` behave identically.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
# NVIDIA GPU? Install torch first with the command from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Running it

Each step depends on the ones before it.

```bash
python src/prepare_data.py     # download a style-balanced sample   (1–2 h, depends on internet; resumable)
python src/embed.py            # CLIP embeddings                    (~1 min Apple GPU / 20–40 min CPU)

python src/show_samples.py     # 16 random paintings, a sanity check (seconds)
python src/reduce.py           # PCA keeping 90% of the variance    (seconds)
python src/cluster.py          # elbow, k-means (k=27), grading      (~1 min)
python src/hierarchy.py        # dendrogram of the styles            (seconds)
python src/tsne_map.py         # 2D map for the app                  (~20 s)
python src/pixel_baseline.py   # same pipeline on raw pixels         (seconds)
python src/predict.py --evaluate   # accuracy of the upload guesses  (~25 s)

streamlit run src/app.py       # style map, upload, clusters, chart explanations
python src/predict.py photo.jpg    # the upload guesses, from the terminal
```

Options: `prepare_data.py --per-style N --max-rows N` (quick test:
`--per-style 5 --max-rows 2000`, then delete `data/`), `reduce.py --variance 0.95`,
`cluster.py --k 20`.

If `prepare_data.py` stops (sleep, network), run it again: it skips the rows it
has already read and carries on.

## The app

`streamlit run src/app.py` opens it at http://localhost:8501. It needs `data/`
(steps 1–2 and `cluster.py`, `tsne_map.py`) and reads the summaries in `results/`.

**The app doesn't show the paintings.** Many are still under copyright and the
dataset gives no right to republish them (see [Dataset](#dataset)), so every
page works from the paintings' labels and CLIP codes instead.

| Page | What it does |
|---|---|
| **Style map** | Every painting as a dot on the t-SNE map, coloured by real style, by the computer's pile or by subject. Highlight a style to see whether it forms its own island, which piles the computer split it into, which styles it looks most like and what its paintings show. With nothing highlighted, the side panel lists the easiest and hardest styles to find. |
| **Upload a picture** | Drop in a painting or a photo of one. Shows your picture, the top three guesses for subject and style with their vote counts, and roughly where it would sit on the map. Warns when the picture doesn't look like a painting. The first upload takes a few seconds while CLIP loads. |
| **Clusters** | Browse the k-means clusters, purest first: which styles each one contains and what its paintings show. |
| **How it works** | Every chart from the analysis, plus tables of the cleanest and most mixed clusters, each with a plain-language note on what it shows, how to read it and what it tells us. |

## What each stage does

**Rule.** `style` is never used to fit anything. It only comes out at the end to
grade the clusters, and in `hierarchy.py`, which describes styles rather than
fitting a model to them.

**Embeddings.** CLIP turns each painting into 512 numbers describing how it
looks; similar-looking paintings get nearby vectors. Rows are scaled to unit
length, so a dot product is the cosine similarity the upload guesses use.

**PCA.** 183 of the 512 dimensions keep 90% of the variance
(`results/pca.json`). Clustering runs on these.

**Clustering.** `cluster.py` fits k-means for k = 5…40 for the elbow plot and
records the ARI at each k, then fits k = 27. ARI peaks around k = 15 (0.200)
and is flat beyond, so k = 27 is chosen for the one-cluster-per-style
comparison rather than for score. `results/clusters.json` has the top three
styles of every cluster and, for every style, how much of it lands in its main
cluster.

**Upload guesses.** `predict.py` embeds the picture with the same CLIP model
(after shrinking it to a 224px thumbnail, like the collection) and lets its 15
most similar paintings vote, weighted by 1 / cosine distance. Unlike the
clustering, this uses the labels. Because the vote always returns an answer,
CLIP is also asked whether the picture looks more like "a painting" or like "a
screenshot / chart / document / photo"; below 30% the app warns that the
guesses are loose. Fewer than 1% of the collection's own paintings fall under
that line, while charts, documents, screenshots, noise and plain colours all
scored under 16%. Uploaded pictures are processed in memory and never saved.

**Raw-pixel baseline.** The same PCA → k-means → ARI on 32×32 grayscale pixels.
It scores close to random, which is the argument for CLIP.

## Figures (`docs/figures/`)

The app's **How it works** page shows all of these with plain-language explanations.

| File | Shows |
|---|---|
| `pca_variance.png` | cumulative explained variance, with the 90% cut |
| `elbow.png` | inertia and ARI against k |
| `crosstab.png` | where each style's paintings end up, ordered so a clean match is the diagonal |
| `dendrogram.png` | which styles look alike |
| `tsne.png` | the 2D map, coloured by cluster and by real style |
| `paintings_*.png` | thumbnails per cluster — **generated locally, not committed** (copyrighted images) |

## Dataset

[`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart): 81,444 paintings, 27 styles, 11 genres, 129 artists.
- The 72 shard files are **not shuffled** (the early ones are missing many styles), so `prepare_data.py` shuffles shard order and caps each style.
- Styles are very unbalanced (Impressionism ≈ 13k paintings, some styles only a few hundred). The cap keeps the clusters from all becoming Impressionism. Four styles have fewer than 400 in the whole dataset: New Realism 314, Synthetic Cubism 216, Analytical Cubism 110, Action Painting 98.
- License: **unknown**, "Data files © Original Authors". The dataset card limits it to **non-commercial research** and asks users to follow [WikiArt.org's terms](https://www.wikiart.org/en/terms-of-use), which give visitors no right to republish the images. Roughly a quarter of this sample (Pop Art, Abstract Expressionism, Minimalism and other mid/late 20th-century styles) is very likely still under copyright. That's why `data/` isn't in git and the app shows paintings as text by default.

## Stretch goals
- NMF on the embeddings: interpretable "visual parts" (course ch. 4)
- Text search with CLIP: type "stormy sea at night" and get matching paintings
- Deploy the app (Streamlit Community Cloud / Hugging Face Spaces)
