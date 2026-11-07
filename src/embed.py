"""Turn every thumbnail into a 512-number CLIP embedding.

Output: data/embeddings.npy, shape (N, 512), with rows in the same order as data/metadata.csv.

    python src/embed.py
"""
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from paths import EMBEDDINGS, IMAGES_DIR, METADATA_CSV

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH = 64


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    return model.to(device).eval(), preprocess


def embed_images(images, model, preprocess, device):
    """PIL images -> (n, 512) float32, unit length so a dot product is the cosine similarity."""
    with torch.no_grad():
        batch = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)
        emb = model.encode_image(batch)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def embed_texts(texts, model, device):
    """Text prompts -> (n, 512) float32, unit length, comparable with image embeddings."""
    tokenizer = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        emb = model.encode_text(tokenizer(texts).to(device))
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def main():
    device = pick_device()
    model, preprocess = load_model(device)

    files = pd.read_csv(METADATA_CSV)["file"].tolist()
    print(f"Embedding {len(files):,} images on {device}...")

    out = []
    for start in range(0, len(files), BATCH):
        images = [Image.open(IMAGES_DIR / f) for f in files[start:start + BATCH]]
        out.append(embed_images(images, model, preprocess, device))
        print(f"  {min(start + BATCH, len(files)):,}/{len(files):,}", end="\r")

    emb = np.concatenate(out)
    assert emb.shape[0] == len(files), "embedding count must match metadata rows"
    np.save(EMBEDDINGS, emb)
    print(f"\nSaved {emb.shape} to {EMBEDDINGS}")


if __name__ == "__main__":
    main()
