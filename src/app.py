"""Streamlit app: explore ~10k paintings clustered by how they look.

Tabs: an interactive painting map (click a dot to open it), an upload that
guesses a picture's subject and style, a "find similar paintings" search, a
cluster browser, and a short explanation with the charts.

Needs data/metadata.csv, data/embeddings.npy (prepare_data.py, embed.py),
data/clusters.npy (cluster.py) and data/tsne.npy (tsne_map.py).

    streamlit run src/app.py
"""
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from paths import (CLUSTERS, CLUSTERS_JSON, EMBEDDINGS, FIGURES_DIR, IMAGES_DIR,
                   METADATA_CSV, PCA_JSON, PIXEL_BASELINE_JSON, TSNE, UPLOAD_ACCURACY_JSON)
from predict import (ALREADY_IN_COLLECTION, ART_PROMPTS, ART_THRESHOLD, K, OTHER_PROMPTS,
                     art_probability, guess, prepare)

st.set_page_config(page_title="Artwork Style Explorer", page_icon=":material/palette:", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  [data-testid="stImage"] img {border-radius: 6px;}
  .small {font-size: 0.82rem; opacity: 0.75; line-height: 1.25;}
  .tag {display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.75rem;
        color: white; margin: 2px 0;}
</style>
""", unsafe_allow_html=True)

# ---------- data ----------

def nice(text):
    """'kuzma-petrov-vodkin' -> 'Kuzma Petrov Vodkin', 'genre_painting' -> 'Genre painting'."""
    if text.startswith("Unknown"):
        return "Unknown"
    if "-" in text:  # artist slugs
        return text.replace("-", " ").title()
    text = text.replace("_", " ")
    return text[0].upper() + text[1:]


@st.cache_data
def load():
    df = pd.read_csv(METADATA_CSV)
    emb = np.load(EMBEDDINGS)
    xy = np.load(TSNE)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df["cluster"] = np.load(CLUSTERS).astype(int)
    df["style_name"] = df["style"].map(nice)
    df["genre_name"] = df["genre"].map(nice)
    df["artist_name"] = df["artist"].map(nice).replace("Unknown", "Unknown artist")
    df["idx"] = np.arange(len(df))
    return df, emb


@st.cache_data
def cluster_summary(df):
    """Per cluster: size, dominant style and its share, and a readable label."""
    rows = []
    for c, g in df.groupby("cluster"):
        counts = g["style_name"].value_counts()
        rows.append({"cluster": c, "size": len(g), "top": counts.index[0],
                     "purity": counts.iloc[0] / len(g)})
    out = pd.DataFrame(rows).sort_values("purity", ascending=False)
    out["label"] = [f"Cluster {r.cluster} · {r.purity:.0%} {r.top} · {r.size} paintings"
                    for r in out.itertuples()]
    return out.set_index("cluster")


@st.cache_data(max_entries=2000)
def square(file, size=224):
    """Centre-cropped square thumbnail, so grids line up whatever the painting's shape."""
    return ImageOps.fit(Image.open(IMAGES_DIR / file).convert("RGB"), (size, size))


@st.cache_resource(show_spinner=False)
def load_clip():
    """CLIP for uploads. Imported lazily so the rest of the app starts without torch."""
    from embed import embed_texts, load_model, pick_device
    device = pick_device()
    model, preprocess = load_model(device)
    return model, preprocess, device, embed_texts(ART_PROMPTS + OTHER_PROMPTS, model, device)


@st.cache_data(max_entries=50, show_spinner=False)
def embed_upload(data):
    """Uploaded bytes -> (CLIP embedding, probability that it looks like art)."""
    from embed import embed_images
    model, preprocess, device, prompts = load_clip()
    q = embed_images([prepare(Image.open(io.BytesIO(data)))], model, preprocess, device)[0]
    return q, art_probability(q, prompts)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


try:
    df, emb = load()
except FileNotFoundError as e:
    st.error(f"Missing `{e.filename}`. Run the steps in the README first.")
    st.stop()

styles = sorted(df["style_name"].unique())
palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
STYLE_COLORS = {s: palette[i % len(palette)] for i, s in enumerate(styles)}
summary = cluster_summary(df)


# ---------- state ----------

if "sel" not in st.session_state:
    st.session_state.sel = int(np.random.default_rng().integers(len(df)))


PAGES = ["Painting map", "Upload a picture", "Find similar", "Clusters", "How it works"]
MAP, UPLOAD, SIMILAR, CLUSTERS_PAGE, ABOUT = PAGES
st.session_state.setdefault("page", MAP)
st.session_state.setdefault("last_page", MAP)


def keep_page():
    """Clicking the active page again would deselect it; stay on it instead."""
    if st.session_state.page is None:
        st.session_state.page = st.session_state.last_page
    st.session_state.last_page = st.session_state.page


def select(i, goto=None):
    st.session_state.sel = int(i)
    if goto:
        st.session_state.page = st.session_state.last_page = goto


def random_painting(pool):
    select(np.random.default_rng().choice(pool))


def similar(i, n, other_styles=False):
    """Nearest paintings by cosine similarity (embeddings are unit length)."""
    with np.errstate(all="ignore"):  # spurious matmul warning on macOS numpy 2.0
        sims = emb @ emb[i]
    order = np.argsort(-sims)
    order = order[order != i]
    if other_styles:
        order = order[df["style"].to_numpy()[order] != df["style"].iat[i]]
    return order[:n], sims


# ---------- small components ----------

def tag(style):
    return f'<span class="tag" style="background:{STYLE_COLORS[style]}">{style}</span>'


def subject(r):
    return "" if r["genre_name"] == "Unknown" else r["genre_name"]


def painting_card(i, sim=None, key="", goto=None):
    r = df.iloc[i]
    st.image(square(r["file"]), width="stretch")
    score = f" · {sim:.2f}" if sim is not None else ""
    st.markdown(f'{tag(r["style_name"])}<div class="small">{r["artist_name"]}{score}</div>',
                unsafe_allow_html=True)
    st.button("Open", key=f"open-{key}-{i}", on_click=select, args=(i, goto),
              icon=":material/arrow_outward:", type="tertiary")


def card_grid(indices, sims=None, cols=4, key="", goto=None):
    for start in range(0, len(indices), cols):
        for col, i in zip(st.columns(cols), indices[start:start + cols]):
            with col:
                painting_card(i, None if sims is None else sims[i], key=key, goto=goto)


def selected_panel(key):
    """The selected painting, its details and its nearest neighbours."""
    i = st.session_state.sel
    r = df.iloc[i]
    st.image(str(IMAGES_DIR / r["file"]), width="stretch")
    st.markdown(f"### {r['artist_name']}")
    st.markdown(f"{tag(r['style_name'])} &nbsp; {subject(r)}", unsafe_allow_html=True)
    s = summary.loc[r["cluster"]]
    st.caption(f"In cluster {r['cluster']}, which is {s.purity:.0%} {s.top}.")
    st.markdown("**Looks most like**")
    near, sims = similar(i, 6)
    for start in range(0, 6, 3):
        for col, j in zip(st.columns(3), near[start:start + 3]):
            with col:
                st.image(square(df["file"].iat[j]), width="stretch")
                st.button(f"{sims[j]:.2f}", key=f"near-{key}-{j}", on_click=select, args=(j,),
                          help=f"{df['style_name'].iat[j]} · open this painting", type="tertiary")


# ---------- header ----------

st.title("Artwork Style Explorer")
st.markdown("Can a computer rediscover art movements **without being told what they are**? "
            "Every painting below was sorted only by how it looks.")

clip = read_json(CLUSTERS_JSON)
pixels = read_json(PIXEL_BASELINE_JSON)
m = st.columns(4)
m[0].metric("Paintings", f"{len(df):,}", border=True)
m[1].metric("Real styles", len(styles), border=True)
m[2].metric("Clusters found", df["cluster"].nunique(), border=True)
if clip and pixels:
    m[3].metric("Match with real styles (ARI)", f"{clip['ari']:.2f}",
                delta=f"{clip['ari'] / pixels['pixels']['ari']:.0f}× raw pixels", border=True,
                help="Adjusted Rand index: 0 = random piles, 1 = exactly the experts' styles. "
                     "Sorting by raw pixel colours scores "
                     f"{pixels['pixels']['ari']:.3f}.")

# A remembered switcher rather than st.tabs: tabs jump back to the first one when an
# upload reruns the script, and this way only the visible page is computed.
page = st.segmented_control("Page", PAGES, key="page", on_change=keep_page,
                            label_visibility="collapsed", width="stretch")
st.write("")


# ---------- upload helpers ----------

def confidence(share):
    return "fairly sure" if share >= 0.6 else "leaning this way" if share >= 0.35 else "unsure"


def guesses(title, table, not_art=False):
    top = table.iloc[0]
    st.markdown(f"<div class='small'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"### {nice(top.label)}")
    sureness = "Closest match only" if not_art else confidence(top.share).capitalize()
    st.caption(f"{sureness} · {top.share:.0%} of the vote")
    for r in table.head(3).itertuples():
        st.progress(float(r.share), text=f"{nice(r.label)} · {r.count} of {K} look-alikes")


def show_upload(data):
    """Guesses, look-alikes and map position for one uploaded picture."""
    try:
        shown_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        shown_img.thumbnail((480, 480))  # keep tall photos from towering over the guesses
        with st.spinner("Looking at your picture… the first one takes a few seconds while the model loads."):
            q, art = embed_upload(data)
    except (UnidentifiedImageError, OSError):
        st.error("Couldn't read that file as an image. Try a JPG or PNG.")
        return

    g = guess(q, emb, df)
    voters = g["voters"]
    top_match = df.iloc[g["nearest"][0]]

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.image(shown_img, width="content")
    with right:
        if art < ART_THRESHOLD:
            st.warning(f"This doesn't look much like a painting (the model rates it {art:.0%} "
                       "\"art\"). The guesses below are just its closest matches, so take them loosely.")
        elif g["best"] >= ALREADY_IN_COLLECTION:
            st.success(f"This looks like a painting that's already in the collection: "
                       f"{top_match['artist_name']}, {top_match['style_name']}.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            guesses("Subject (genre)", g["genre"], not_art=art < ART_THRESHOLD)
        with c2:
            guesses("Art style", g["style"], not_art=art < ART_THRESHOLD)
        cluster = int(df["cluster"].iloc[voters].mode()[0])
        s = summary.loc[cluster]
        st.caption(f"Its look-alikes mostly sit in cluster {cluster}, which is {s.purity:.0%} {s.top}. "
                   f"Closest match similarity: {g['best']:.2f}.")

    st.divider()
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown("#### Paintings it looks most like")
        st.caption("“Open” shows that painting on the map.")
        card_grid(g["nearest"][:6], g["sims"], cols=3, key="up", goto=MAP)
    with right:
        st.markdown("#### Where it would sit on the map")
        w = np.clip(g["sims"][voters], 0, None)
        ux, uy = np.average(df["x"].iloc[voters], weights=w), np.average(df["y"].iloc[voters], weights=w)
        mini = px.scatter(df, x="x", y="y", color="style_name", color_discrete_map=STYLE_COLORS,
                          hover_data={"x": False, "y": False}, labels={"style_name": "Style"},
                          render_mode="webgl", height=460)
        mini.update_traces(marker=dict(size=3, opacity=0.25))
        mini.add_trace(go.Scattergl(x=[ux], y=[uy], mode="markers", name="Your picture",
                                    hoverinfo="name", marker=dict(size=22, symbol="star", color="gold",
                                    line=dict(width=1.5, color="black"))))
        mini.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False,
                           margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(mini, config={"displayModeBar": False})
        st.caption(f"Placed among its {K} closest look-alikes. The map can't be redrawn for new "
                   "pictures, so this spot is approximate.")


# ---------- map ----------

if page == MAP:
    left, right = st.columns([3, 1.1], gap="large")
    with left:
        c1, c2 = st.columns([1, 2])
        color_by = c1.segmented_control("Colour by", ["Style", "Cluster", "Genre"], default="Style")
        shown = c2.multiselect("Show only these styles", styles, placeholder="All styles")
        view = df[df["style_name"].isin(shown)] if shown else df

        col = {"Style": "style_name", "Cluster": "cluster", "Genre": "genre_name"}[color_by or "Style"]
        plot = view.assign(cluster=view["cluster"].astype(str))
        fig = px.scatter(
            plot, x="x", y="y", color=col, custom_data=["idx"],
            color_discrete_map=STYLE_COLORS if col == "style_name" else None,
            category_orders={"cluster": [str(c) for c in sorted(df["cluster"].unique())],
                             "style_name": styles, "genre_name": sorted(df["genre_name"].unique())},
            hover_data={"x": False, "y": False, "style_name": True, "artist_name": True,
                        "genre_name": True, "cluster": True},
            labels={"style_name": "Style", "artist_name": "Artist", "genre_name": "Genre",
                    "cluster": "Cluster"},
            render_mode="webgl", height=680)
        # keep every dot fully visible after a click (plotly dims unselected points by default)
        fig.update_traces(marker=dict(size=5, opacity=0.75), unselected=dict(marker=dict(opacity=0.75)))
        sel = df.iloc[st.session_state.sel]
        fig.add_trace(go.Scattergl(x=[sel.x], y=[sel.y], mode="markers", name="Selected",
                                   hoverinfo="skip", marker=dict(size=18, symbol="star",
                                   color="gold", line=dict(width=1.5, color="black"))))
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(title=None, itemsizing="constant", font=dict(size=11)),
                          dragmode="pan")
        event = st.plotly_chart(fig, key="map", on_select="rerun", selection_mode="points",
                                config={"scrollZoom": False, "displaylogo": False})
        st.caption("Each dot is a painting; similar-looking paintings sit close together. "
                   "**Click a dot** to open it. Drag to pan, use the + / − buttons to zoom, double-click to reset.")

        # Only react to a *new* click, so other buttons can still change the selection.
        points = event.selection.points if event else []
        pick = points[0]["customdata"][0] if points else None
        if pick is not None and pick != st.session_state.get("last_map_pick"):
            st.session_state.last_map_pick = pick
            select(pick)
            st.rerun()

    with right:
        st.button("Surprise me", icon=":material/shuffle:", on_click=random_painting,
                  args=(view["idx"].to_numpy(),), width="stretch")
        selected_panel("map")


# ---------- upload ----------

elif page == UPLOAD:
    st.markdown("Upload a painting, or a photo of one, and the app guesses its **subject** and "
                "**art style** from the paintings in the collection it looks most like.")
    acc = read_json(UPLOAD_ACCURACY_JSON)
    if acc:
        st.caption(f"Tested on the collection, the first guess is right {acc['genre']['top1']:.0%} of the "
                   f"time for subject and {acc['style']['top1']:.0%} for style; the right answer is in the "
                   f"top three {acc['genre']['top3']:.0%} and {acc['style']['top3']:.0%} of the time. "
                   "Your picture is only used on this computer and isn't saved.")
    file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png", "webp"],
                            label_visibility="collapsed")
    if file is None:
        st.info("Tip: photograph the painting straight on, with as little wall around it as possible.",
                icon=":material/photo_camera:")
    else:
        show_upload(file.getvalue())


# ---------- find similar ----------

elif page == SIMILAR:
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("#### Pick a painting")
        pick_style = st.selectbox("Style", ["Any style"] + styles)
        pool = df if pick_style == "Any style" else df[df["style_name"] == pick_style]
        st.button("Random painting", icon=":material/shuffle:", on_click=random_painting,
                  args=(pool["idx"].to_numpy(),), width="stretch", type="primary")
        i = st.session_state.sel
        r = df.iloc[i]
        st.image(str(IMAGES_DIR / r["file"]), width="stretch")
        st.markdown(f"**{r['artist_name']}**  \n{tag(r['style_name'])} {subject(r)}",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Paintings that look like it")
        other = st.toggle("Only show other styles",
                          help="Find look-alikes across art movements")
        near, sims = similar(st.session_state.sel, 12, other_styles=other)
        if not other:
            same = (df["style"].to_numpy()[near] == r["style"]).mean()
            st.caption(f"{same:.0%} of these are the same style ({r['style_name']}). "
                       "The number is the similarity: 1.00 would be identical.")
        card_grid(near, sims, cols=4, key="sim")


# ---------- clusters ----------

elif page == CLUSTERS_PAGE:
    st.markdown("The computer made these piles on its own. Each one is labelled with the real "
                "style that appears in it most — a high percentage means it rediscovered that style.")
    c = st.selectbox("Cluster (purest first)", summary.index,
                     format_func=lambda c: summary.loc[c, "label"])
    members = df[df["cluster"] == c]
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        comp = members["style_name"].value_counts().head(8).rename_axis("style").reset_index()
        comp["share"] = comp["count"] / len(members)
        comp = comp[comp["share"] >= 0.02]
        bar = px.bar(comp[::-1], x="share", y="style", orientation="h", color="style",
                     color_discrete_map=STYLE_COLORS, text_auto=".0%", height=90 + 38 * len(comp))
        bar.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_title=None,
                          yaxis_title=None, margin=dict(l=0, r=0, t=30, b=0),
                          title=dict(text="What's inside", font=dict(size=14)))
        st.plotly_chart(bar, config={"displayModeBar": False})
        genres = members["genre_name"].value_counts()
        st.caption("Most common subjects: " + ", ".join(
            f"{g} ({n})" for g, n in genres.head(3).items()))
    with right:
        st.session_state.setdefault("cluster_seed", 0)
        if st.button("Show other paintings", icon=":material/refresh:"):
            st.session_state.cluster_seed += 1
        sample = members.sample(min(12, len(members)), random_state=st.session_state.cluster_seed)
        card_grid(sample["idx"].tolist(), cols=6, key="clu", goto=MAP)


# ---------- about ----------

elif page == ABOUT:
    pca = read_json(PCA_JSON) or {}
    upload_acc = read_json(UPLOAD_ACCURACY_JSON)
    n_comp = pca.get("n_components", 183)
    half = pca.get("components_for", {}).get("50%", 23)
    ari = clip["ari"] if clip else 0.19
    pix_ari = pixels["pixels"]["ari"] if pixels else 0.02

    def explain(n, title, what, how, learned, image=None, fig=None, local_only=False, side=True):
        """One plot in a bordered box with three short plain-language notes: beside the plot
        for normal-sized charts (side=True), underneath it for wide ones."""
        notes = [("What you're looking at", what), ("How to read it", how),
                 ("What it tells us", learned)]
        with st.container(border=True):
            st.markdown(f"### {n}. {title}")
            plot_area, note_areas = (st.columns([1.45, 1], gap="large") if side else (st.container(), None))
            with plot_area:
                if image:
                    path = FIGURES_DIR / image
                    if path.exists():
                        # "stretch" collapses to an icon inside a plain container; use a fixed width there
                        st.image(str(path), width="stretch" if side else 1150)
                    else:
                        script = "show_samples.py" if "random" in image else "cluster.py"
                        st.info(f"This picture isn't here yet. Run `python src/{script}` to create it"
                                + (" (it isn't stored in git because the paintings are copyrighted)."
                                   if local_only else "."))
                if fig is not None:
                    st.plotly_chart(fig, config={"displayModeBar": False})
            if side:
                with note_areas:
                    for head, text in notes:
                        st.markdown(f"**{head}**\n\n{text}")
            else:
                for col, (head, text) in zip(st.columns(3, gap="large"), notes):
                    col.markdown(f"**{head}**\n\n{text}")

    st.markdown(f"""
## How it works, one chart at a time
The big question: **if a computer sorts {len(df):,} paintings only by how they look, does it
rediscover the art movements that experts use?** Each box below is one step, with a chart and
three short notes: what it shows, how to read it, and what we learned.
""")
    with st.expander("Words used on this page"):
        st.markdown("""
- **Style**: the art movement experts put a painting in (Cubism, Impressionism, …). There are 27.
- **Genre / subject**: what the painting shows (portrait, landscape, still life, …).
- **CLIP**: an AI model that turns a picture into a list of 512 numbers, a kind of *code for how it looks*. Pictures that look alike get similar codes.
- **Cluster / pile**: a group of paintings the computer put together because their codes are similar. It is never told the styles.
- **ARI (adjusted Rand index)**: a score for how well the computer's piles match the experts' styles. **0** = no better than random piles, **1** = exactly the experts' styles.
""")

    explain(1, "The paintings", image="paintings_random.png", local_only=True,
            what="16 paintings picked at random from the collection. The title above each is the "
                 "style the experts gave it.",
            how="Just look. It's a quick check that every picture really matches its label "
                "before anything is measured.",
            learned="The collection is very varied: portraits, landscapes, abstract shapes, Japanese "
                    "prints. Some styles are easy to spot, while others (like Impressionism and "
                    "Post-Impressionism) are hard to tell apart even for people.")

    explain(2, "Squeezing 512 numbers into fewer", image="pca_variance.png",
            what="CLIP describes each painting with 512 numbers, but many of them repeat the same "
                 "information. **PCA** rewrites them as *summary numbers*, the most useful first.",
            how="Left to right: how many summary numbers we keep. Bottom to top: how much of the "
                "difference between paintings they still capture (1.0 = all of it). The curve shoots "
                "up and then flattens, so the first few numbers carry most of the information.",
            learned=f"Just {half} numbers keep half of the information, and **{n_comp} keep 90%** "
                    "(the dashed lines). We sort with those instead of all 512, which is faster and "
                    "ignores tiny details that are mostly noise.")

    explain(3, "How many piles?", image="elbow.png",
            what="The sorting method (**k-means**) has to be told how many piles to make. We tried "
                 "5, 10, 15 … 40 piles and scored each try twice.",
            how="**Blue** (left scale): how tightly packed the piles are; lower is tighter. "
                "**Orange** (right scale): how well the piles match the real styles (ARI). The usual "
                "trick is to look for an *elbow*, where the blue line suddenly stops dropping.",
            learned="There is **no clear elbow**: the blue line keeps sliding down. Paintings don't "
                    "fall into neat separate groups, because art styles blend into each other. The "
                    "match with real styles peaks around 15 piles (0.20). We use **27 piles** (dotted "
                    "line), one per real style, so the comparison is fair.")

    explain(4, "Did the piles find the styles?", image="crosstab.png", side=False,
            what="Every **row** is a real style; every **column** is one of the computer's piles. "
                 "Each square's colour shows what share of that style's paintings landed in that pile.",
            how="A **dark square** means most of that style ended up in one pile, so it was found "
                "cleanly. A row of **pale squares** means the style was scattered over many piles. "
                "Columns are ordered so clean matches line up along a diagonal.",
            learned=f"**Ukiyo-e** and the **Cubisms** give dark squares; the three Cubisms even share "
                    f"one pile. The Renaissance styles, Expressionism, Fauvism and Symbolism are "
                    f"smeared across many piles. Overall score: **ARI {ari:.2f}**, clearly better "
                    "than random (0) but far from perfect (1).")

    explain(5, "The cleanest piles", image="paintings_purest_clusters.png", side=False,
            local_only=True,
            what="The 6 piles that are most dominated by a single style, with 8 random paintings "
                 "from each (one pile per row).",
            how="The first line above each picture is the pile number, its main style and how much "
                "of the pile that style makes up. The second line is that painting's real style.",
            learned="The Japanese-print pile is **99% Ukiyo-e**. But look at the Rococo pile: it is "
                    "really a pile of **formal portraits**, Rococo and Romanticism mixed "
                    "together. The computer grouped them by *what is painted*, not by art movement.")

    explain(6, "The most mixed piles", image="paintings_mixed_clusters.png", side=False,
            local_only=True,
            what="The 6 piles where no style dominates, again with 8 paintings each.",
            how="Same layout as above. The percentages are low, so the experts would scatter "
                "these paintings across many styles.",
            learned="These piles are **not random**. Most have a clear theme: **portraits**, **nudes**, "
                    "**pencil and ink drawings**, **buildings and streets**. The computer found real "
                    "groups; they just aren't the groups art historians use. That's the main reason "
                    "the score in step 4 isn't higher.")

    explain(7, "A family tree of styles", image="dendrogram.png",
            what="We averaged the code of every painting in each style, then joined the most alike "
                 "styles together, step by step, into a family tree (**hierarchical clustering**).",
            how="Read from **right to left**. Styles joined by a short line near the right edge look "
                "very alike; the further left two branches meet, the more different they are. The "
                "colours mark the main branches.",
            learned="It **rebuilds art history without knowing any dates**: the four Renaissance "
                    "styles sit together, Baroque pairs with Rococo, the three Cubisms join up, "
                    "Impressionism sits next to Post-Impressionism, and the abstract movements get a "
                    "branch of their own. Ukiyo-e stands alone, looking like nothing else.")

    explain(8, "A map of every painting", image="tsne.png", side=False,
            what=f"**t-SNE** squashes each painting's {n_comp} numbers down to just 2, so every "
                 "painting can be drawn as a dot. Similar paintings land close together. It's the "
                 "same map as on the Painting map page.",
            how="**Left**: dots coloured by the computer's piles. **Right**: the same dots coloured "
                "by the real styles. A tight island of one colour on the right means that style has a "
                "look of its own. Only *closeness* matters; the distance between far-apart islands "
                "doesn't mean much.",
            learned="Ukiyo-e is a separate island at the bottom, and the abstract styles gather on "
                    "the left. The middle is one big mixed crowd: the many 19th- and "
                    "20th-century styles of painted people and places, which blend into each other.")

    bars = px.bar(x=["Raw pixel colours", "CLIP codes"], y=[pix_ari, ari], text=[f"{pix_ari:.2f}", f"{ari:.2f}"],
                  color=["Raw pixel colours", "CLIP codes"],
                  color_discrete_map={"Raw pixel colours": "#9aa0a6", "CLIP codes": "#1f77b4"}, height=340)
    bars.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Match with real styles (ARI)",
                       margin=dict(l=0, r=0, t=10, b=0))
    bars.update_traces(textposition="outside", width=0.45)
    explain(9, "Why use CLIP at all?", fig=bars,
            what="The same sorting done twice: once with CLIP's codes, once with the plain pixel "
                 "colours of each painting (shrunk to 32×32 grey pixels).",
            how="Taller bar = piles that match the real styles better (ARI). 0 would be random "
                "piles; 1 would be a perfect match.",
            learned=f"Pixel colours score **{pix_ari:.2f}**, about the same as random. CLIP scores "
                    f"**{ari:.2f}**, roughly **{ari / pix_ari:.0f}× better**. Light and dark alone say "
                    "little about style; CLIP picks up shapes, brushwork and subjects.")

    if upload_acc:
        rows = []
        for key, name in [("genre", "Subject"), ("style", "Style")]:
            r = upload_acc[key]
            rows += [{"what": name, "measure": "Random guessing", "value": r["random_guess"]},
                     {"what": name, "measure": "First guess right", "value": r["top1"]},
                     {"what": name, "measure": "Right answer in top 3", "value": r["top3"]}]
        acc_fig = px.bar(pd.DataFrame(rows), x="what", y="value", color="measure", barmode="group",
                         text_auto=".0%", height=360,
                         color_discrete_map={"Random guessing": "#c4c7cc", "First guess right": "#1f77b4",
                                             "Right answer in top 3": "#9ecae1"})
        acc_fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.08], xaxis_title=None,
                              yaxis_title=None, legend_title=None, margin=dict(l=0, r=0, t=10, b=0))
        g, s_ = upload_acc["genre"], upload_acc["style"]
        explain(10, "How good are the upload guesses?", fig=acc_fig,
                what="A test of the **Upload a picture** page. We hid each painting in turn and let its "
                     f"{upload_acc['k']} most similar paintings vote on its subject and style.",
                how="**Grey**: what random guessing would get. **Dark blue**: how often the first "
                    "guess is right. **Light blue**: how often the right answer is among the top three.",
                learned=f"Subject: first guess right **{g['top1']:.0%}** of the time ({g['top3']:.0%} in "
                        f"the top 3). Style: **{s_['top1']:.0%}** ({s_['top3']:.0%}). Style is harder "
                        "because neighbouring movements look alike, the same reason as in steps 3–4. "
                        "Unlike everything above, this step **does** use the experts' labels.")
