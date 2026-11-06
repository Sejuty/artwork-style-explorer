"""Turn every thumbnail into a 512-number CLIP embedding.

Output: data/embeddings.npy, shape (N, 512), with rows in the same order as data/metadata.csv.

    python embed.py
"""
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from prepare_data import DATA, IMAGES, META

BATCH = 64


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()

    files = pd.read_csv(META)["file"].tolist()
    print(f"Embedding {len(files):,} images on {device}...")

    out = []
    with torch.no_grad():
        for start in range(0, len(files), BATCH):
            batch = torch.stack([preprocess(Image.open(IMAGES / f).convert("RGB"))
                                 for f in files[start:start + BATCH]]).to(device)
            emb = model.encode_image(batch)
            out.append((emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy())  # unit length: dot product = cosine similarity
            print(f"  {min(start + BATCH, len(files)):,}/{len(files):,}", end="\r")

    emb = np.concatenate(out).astype(np.float32)
    assert emb.shape[0] == len(files), "embedding count must match metadata rows"
    np.save(DATA / "embeddings.npy", emb)
    print(f"\nSaved {emb.shape} to {DATA / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
