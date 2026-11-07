"""Look at the data: a 4x4 grid of random paintings with their style.

A quick check that images and labels line up before any analysis.

    python src/show_samples.py
"""
from common import SEED, load_metadata, require, thumbnail_grid
from paths import FIGURES_DIR, METADATA_CSV, ensure_dirs


def main():
    require(METADATA_CSV)
    ensure_dirs()
    df = load_metadata()
    sample = df.sample(16, random_state=SEED)
    out = FIGURES_DIR / "paintings_random.png"
    thumbnail_grid(sample, 4, 4, sample["style"].str.replace("_", " "), out,
                   suptitle="16 random paintings")
    print(f"{len(df):,} paintings, {df['style'].nunique()} styles")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
