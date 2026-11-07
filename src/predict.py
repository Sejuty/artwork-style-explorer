"""Guess the genre (subject) and style of a new picture from its look-alikes.

The picture gets the same CLIP embedding as the collection, and its K most
similar paintings vote, weighted by how close each one is (k-nearest
neighbours). Unlike the clustering, this *does* use the labels: it is a
feature of the app, not part of the "rediscover styles unsupervised" question.

    python src/predict.py photo.jpg [more.jpg ...]   # print guesses
    python src/predict.py --evaluate                  # measure accuracy -> results/upload_accuracy.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from paths import EMBEDDINGS, METADATA_CSV, UPLOAD_ACCURACY_JSON, ensure_dirs

K = 15
THUMBNAIL = 224   # the collection was embedded from 224px thumbnails (prepare_data.py)
ALREADY_IN_COLLECTION = 0.97   # best similarity above this: (nearly) the same picture

# "Is this a painting at all?" CLIP compares the picture with these descriptions.
# The nearest-neighbour vote always answers, even for a screenshot, so the app
# warns when the art descriptions win less than ART_THRESHOLD of the probability.
# On the collection itself fewer than 1% of paintings fall below 0.3; charts,
# documents, screenshots, noise and plain colours all scored under 0.16.
ART_PROMPTS = ["a painting", "a photo of a painting", "an artwork", "a drawing", "a print",
               "an abstract painting", "a sketch"]
OTHER_PROMPTS = ["a photo", "a screenshot", "a document with text", "a chart or graph",
                 "random noise", "a selfie", "a website", "a plain colour background"]
ART_THRESHOLD = 0.3


def prepare(img):
    """Match how the collection's images were made: upright, RGB, 224px thumbnail."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((THUMBNAIL, THUMBNAIL))
    return img


def vote(sims, labels, k=K, exclude=None):
    """Weighted vote of the k most similar rows.

    sims: similarity of the query to every painting. labels: one label per
    painting; rows whose label is missing (None/NaN) don't vote. Weights are
    1 / cosine distance, the same as sklearn's KNeighborsClassifier(weights="distance").
    Returns (table with label/share/count, the voters' indices).
    """
    ok = pd.notna(labels)
    if exclude is not None:
        ok &= ~exclude
    candidates = np.flatnonzero(ok)
    voters = candidates[np.argsort(-sims[candidates])[:k]]
    weights = 1.0 / np.maximum(1.0 - sims[voters], 1e-6)
    table = (pd.DataFrame({"label": labels[voters], "weight": weights})
             .groupby("label")["weight"].agg(["sum", "size"])
             .rename(columns={"size": "count"}))
    table["share"] = table["sum"] / table["sum"].sum()
    table = table.sort_values("share", ascending=False).reset_index()
    return table[["label", "share", "count"]], voters


def art_probability(query, prompt_emb):
    """Share of CLIP's probability that goes to the ART_PROMPTS (prompt_emb = embed_texts(ART_PROMPTS + OTHER_PROMPTS))."""
    logits = 100.0 * prompt_emb @ query   # CLIP's own logit scale
    p = np.exp(logits - logits.max())
    return float(p[:len(ART_PROMPTS)].sum() / p.sum())


def genre_labels(df):
    """Genre per painting, with the unknown ones left out of the vote."""
    return df["genre"].where(~df["genre"].str.startswith("Unknown")).to_numpy(dtype=object)


def guess(query, emb, df, k=K):
    """Everything the app shows for one uploaded picture."""
    with np.errstate(all="ignore"):  # spurious matmul warning on macOS numpy 2.0
        sims = emb @ query
    style, style_voters = vote(sims, df["style"].to_numpy(dtype=object), k)
    genre, _ = vote(sims, genre_labels(df), k)
    order = np.argsort(-sims)
    return {"style": style, "genre": genre, "sims": sims, "nearest": order,
            "voters": style_voters, "best": float(sims[order[0]])}


def evaluate(df, emb, k=K):
    """Hide each painting in turn and let the rest vote on it (leave-one-out)."""
    labels = {"style": df["style"].to_numpy(dtype=object), "genre": genre_labels(df)}
    hits = {t: {"top1": 0, "top3": 0, "n": 0} for t in labels}
    for start in range(0, len(df), 1000):
        with np.errstate(all="ignore"):
            block = emb[start:start + 1000] @ emb.T
        for row, sims in enumerate(block):
            i = start + row
            self_mask = np.zeros(len(df), bool)
            self_mask[i] = True
            for target, y in labels.items():
                if pd.isna(y[i]):
                    continue
                table, _ = vote(sims, y, k, exclude=self_mask)
                top = table["label"].tolist()
                hits[target]["n"] += 1
                hits[target]["top1"] += top[0] == y[i]
                hits[target]["top3"] += y[i] in top[:3]
    out = {"k": k}
    for target, h in hits.items():
        classes = pd.Series(labels[target]).dropna().nunique()
        out[target] = {"paintings": h["n"], "classes": int(classes),
                       "top1": round(h["top1"] / h["n"], 4), "top3": round(h["top3"] / h["n"], 4),
                       "random_guess": round(1 / classes, 4)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", help="pictures to guess")
    p.add_argument("--evaluate", action="store_true", help="measure accuracy on the collection")
    args = p.parse_args()
    if not args.images and not args.evaluate:
        p.error("give one or more images, or --evaluate")

    df = pd.read_csv(METADATA_CSV)
    emb = np.load(EMBEDDINGS)

    if args.evaluate:
        ensure_dirs()
        result = evaluate(df, emb)
        UPLOAD_ACCURACY_JSON.write_text(json.dumps(result, indent=2))
        for t in ("genre", "style"):
            r = result[t]
            print(f"{t:5s}: top-1 {r['top1']:.0%}  top-3 {r['top3']:.0%}  "
                  f"(random {r['random_guess']:.0%}, {r['classes']} classes, {r['paintings']:,} paintings)")
        print(f"Saved {UPLOAD_ACCURACY_JSON}")

    if args.images:
        from embed import embed_images, embed_texts, load_model, pick_device  # torch only when needed
        device = pick_device()
        model, preprocess = load_model(device)
        prompts = embed_texts(ART_PROMPTS + OTHER_PROMPTS, model, device)
        for path in args.images:
            q = embed_images([prepare(Image.open(path))], model, preprocess, device)[0]
            g = guess(q, emb, df)
            art = art_probability(q, prompts)
            note = "" if art >= ART_THRESHOLD else "  <- doesn't look like a painting"
            print(f"\n{path}  (looks like art {art:.0%}, best match similarity {g['best']:.2f}){note}")
            for t in ("genre", "style"):
                print(f"  {t}: " + ", ".join(f"{r.label} {r.share:.0%} ({r.count}/{K})"
                                             for r in g[t].head(3).itertuples()))


if __name__ == "__main__":
    main()
