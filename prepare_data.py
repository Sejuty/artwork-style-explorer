"""Stream WikiArt from Hugging Face and keep a style-balanced sample of thumbnails.

Nothing big is stored: the 34 GB dataset is read once, and only small JPEGs are saved.
Output: data/images/<idx>.jpg and data/metadata.csv

    python prepare_data.py                      # ~10k images, 400 per style
    python prepare_data.py --per-style 5 --max-rows 2000   # quick smoke test
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATA = Path(__file__).parent / "data"
IMAGES = DATA / "images"
META = DATA / "metadata.csv"
FIELDS = ["file", "style", "genre", "artist"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-style", type=int, default=400, help="max paintings kept per style")
    p.add_argument("--size", type=int, default=224, help="thumbnail size in pixels (CLIP uses 224)")
    p.add_argument("--max-rows", type=int, default=None, help="stop after reading this many rows (for testing)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)

    # Resume: count what earlier runs already kept, so a restart doesn't start over.
    kept = Counter()
    if META.exists():
        with META.open() as f:
            for row in csv.DictReader(f):
                kept[row["style"]] += 1
    next_idx = sum(kept.values())

    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    names = {col: ds.features[col].int2str for col in ("style", "genre", "artist")}
    n_styles = ds.features["style"].num_classes
    # The 72 shard files are sorted (early ones lack many styles), so shuffle shard order + a row buffer.
    ds = ds.shuffle(seed=args.seed, buffer_size=1000)

    new_file = not META.exists()
    with META.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for i, ex in enumerate(ds):
            if args.max_rows and i >= args.max_rows:
                break
            style = names["style"](ex["style"])
            if kept[style] >= args.per_style:
                continue

            img = ex["image"].convert("RGB")
            img.thumbnail((args.size, args.size))  # keeps aspect ratio; CLIP's preprocessing crops later
            fname = f"{next_idx:06d}.jpg"
            img.save(IMAGES / fname, quality=90)
            writer.writerow({
                "file": fname,
                "style": style,
                "genre": names["genre"](ex["genre"]),
                "artist": names["artist"](ex["artist"]),
            })
            f.flush()  # so an interrupted run can resume cleanly
            kept[style] += 1
            next_idx += 1

            if next_idx % 200 == 0:
                print(f"read {i + 1:,} rows | kept {next_idx:,} | styles full: "
                      f"{sum(c >= args.per_style for c in kept.values())}/{n_styles}")
            if len(kept) == n_styles and all(c >= args.per_style for c in kept.values()):
                break  # every style is full: no need to read the rest

    print(f"\nDone. {next_idx:,} images in {IMAGES}")
    for style, c in sorted(kept.items(), key=lambda x: -x[1]):
        print(f"  {style:30s} {c}")


if __name__ == "__main__":
    main()
