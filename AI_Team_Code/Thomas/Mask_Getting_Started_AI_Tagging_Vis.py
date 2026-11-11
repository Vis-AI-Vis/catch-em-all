import io
import os
import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for Windows/VS Code
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Always resolve paths relative to this script's folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(SCRIPT_DIR, "phase1_mask_analysis.csv")
OUT_CSV_PATH = os.path.join(SCRIPT_DIR, "phase1_mask_with_animals.csv")
FORCE_RECOMPUTE_CLIP = False  # Set True to force re-running CLIP even if OUT_CSV_PATH exists


def safe_savefig(fig, filename):
    """Safely saves a matplotlib figure on Windows, avoiding OneDrive/VS Code file-lock Errno 22."""
    out_path = os.path.join(SCRIPT_DIR, filename)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    plt.close(fig)

    try:
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"--> Saved {out_path}")
    except OSError:
        alt_path = os.path.join(SCRIPT_DIR, filename.replace(".png", "_unlocked.png"))
        with open(alt_path, "wb") as f:
            f.write(data)
        print(f"--> [File was locked by Windows/VS Code] Saved instead to: {alt_path}")


# Official 20 AM-2k animal categories from Li et al. (IJCV 2022)
AM2K_CATEGORIES = [
    "alpaca", "antelope", "bear", "camel", "cat",
    "cattle", "deer", "dog", "elephant", "giraffe",
    "horse", "kangaroo", "leopard", "lion", "monkey",
    "rabbit", "rhinoceros", "sheep", "tiger", "zebra"
]


def extract_clip_text_embeds(model, inputs):
    """Safely extracts normalized 512-dim text embeddings across all transformers versions."""
    out = model.get_text_features(**inputs)
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
    if hasattr(model, "text_projection") and out.shape[-1] != model.text_projection.out_features:
        out = model.text_projection(out)
    return F.normalize(out, p=2, dim=-1)


def extract_clip_image_embeds(model, inputs):
    """Safely extracts normalized 512-dim image embeddings across all transformers versions."""
    out = model.get_image_features(**inputs)
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "image_embeds", getattr(out, "pooler_output", out[0]))
    if hasattr(model, "visual_projection") and out.shape[-1] != model.visual_projection.out_features:
        out = model.visual_projection(out)
    return F.normalize(out, p=2, dim=-1)


# ==============================================================================
# 1-4. Load Cached Enriched CSV if Available, Else Run CLIP + t-SNE
# ==============================================================================
dist_matrix = None

if os.path.exists(OUT_CSV_PATH) and not FORCE_RECOMPUTE_CLIP:
    print(f"--> Loading already-computed animal labels & t-SNE from {OUT_CSV_PATH}...")
    df = pd.read_csv(OUT_CSV_PATH)
    coords_2d = df[["tsne_x", "tsne_y"]].to_numpy()
    dist_matrix = euclidean_distances(coords_2d)
else:
    if not os.path.exists(CSV_PATH):
        CSV_PATH = "phase1_mask_analysis.csv"
    df = pd.read_csv(CSV_PATH)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--> Running CLIP animal classification on {device} (Masks)...")

    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    clip_model.eval()

    # Specialized prompt for binary mask / silhouette representations
    text_prompts = [f"a black and white silhouette mask of a {animal} on a black background" for animal in AM2K_CATEGORIES]
    text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = extract_clip_text_embeds(clip_model, text_inputs)

    BATCH_SIZE = 32
    animal_labels = []
    animal_confs = []
    true_fg_coverages = []
    true_fg_saturations = []
    true_fg_brightnesses = []
    image_embeddings = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Tagging 20 AM-2k Animals (Mask)"):
        batch_paths = df["filepath"].iloc[i : i + BATCH_SIZE].tolist()
        batch_imgs = []

        for p in batch_paths:
            clean_p = p.replace("\\", "/")
            if not os.path.exists(clean_p):
                clean_p = os.path.join(SCRIPT_DIR, "mask", os.path.basename(p))
            if not os.path.exists(clean_p):
                clean_p = os.path.join("./mask", os.path.basename(p))

            raw_mask = Image.open(clean_p).convert("L")
            img_rgb = raw_mask.convert("RGB")
            batch_imgs.append(img_rgb)

            # Mask itself is the target
            mask_arr = np.array(raw_mask)
            fg_mask = mask_arr > 128
            true_fg_coverages.append(round(float(np.mean(fg_mask)), 4))

            # Masks are binary (saturation is 0, foreground brightness is ~1.0)
            true_fg_saturations.append(0.0)
            true_fg_brightnesses.append(round(float(mask_arr[fg_mask].mean() / 255.0), 4) if np.any(fg_mask) else 0.0)

        img_inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = extract_clip_image_embeds(clip_model, img_inputs)
            logits_per_image = (img_feats @ text_features.T) * 100.0
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            image_embeddings.append(img_feats.cpu().numpy())

        for prob_vec in probs:
            top_idx = int(np.argmax(prob_vec))
            animal_labels.append(AM2K_CATEGORIES[top_idx])
            animal_confs.append(round(float(prob_vec[top_idx]), 4))

    df["animal_category"] = animal_labels
    df["animal_clip_conf"] = animal_confs
    df["fg_coverage"] = true_fg_coverages
    df["fg_saturation_only"] = true_fg_saturations
    df["fg_brightness_only"] = true_fg_brightnesses

    all_embeds = np.vstack(image_embeddings)
    dist_matrix = cosine_distances(all_embeds)
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42, perplexity=30)
    coords = tsne.fit_transform(dist_matrix)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"--> Saved enriched CSV with animal labels & 2D coordinates to {OUT_CSV_PATH}")

# ==============================================================================
# 5. Plot 1: Bipartite Bubble Matrix
# ==============================================================================
top15_pokes = df["top1_pokemon"].value_counts().head(15).index.tolist()

pivot_count = pd.crosstab(df["animal_category"], df["top1_pokemon"]).reindex(
    index=AM2K_CATEGORIES, columns=top15_pokes, fill_value=0
)
animal_totals = df["animal_category"].value_counts().reindex(AM2K_CATEGORIES, fill_value=1)
pivot_pct = pivot_count.div(animal_totals, axis=0) * 100.0

pivot_ent = df.pivot_table(
    index="animal_category", columns="top1_pokemon", values="normalized_entropy", aggfunc="mean"
).reindex(index=AM2K_CATEGORIES, columns=top15_pokes)

fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
x_coords, y_coords, sizes, colors = [], [], [], []

for y_idx, animal in enumerate(AM2K_CATEGORIES):
    for x_idx, poke in enumerate(top15_pokes):
        pct = pivot_pct.loc[animal, poke]
        ent = pivot_ent.loc[animal, poke]
        if pct > 1.0 and not np.isnan(ent):
            x_coords.append(x_idx)
            y_coords.append(y_idx)
            sizes.append(pct * 12)
            colors.append(ent)

scatter = ax.scatter(
    x_coords, y_coords, s=sizes, c=colors, cmap="viridis",
    alpha=0.9, edgecolors="black", linewidth=1.0, vmin=0.1, vmax=0.85
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(
    "Mean Normalized Entropy (0.1 Dark Purple = High-Confidence Trap | 0.8 Yellow = Uncertain Sink)",
    fontsize=10, fontweight="bold"
)

ax.set_xticks(range(len(top15_pokes)))
ax.set_xticklabels(top15_pokes, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(AM2K_CATEGORIES)))
ax.set_yticklabels(AM2K_CATEGORIES, fontsize=10)
ax.set_title(
    "Mask Silhouette Animal-to-Pokémon OOD Confusion Matrix\n"
    "(Bubble Size = % of Animal Assigned to Pokémon | Dark Purple = Low Entropy / High Confidence)",
    fontsize=13, fontweight="bold"
)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
safe_savefig(fig, "phase1_mask_animal_pokemon_bubble_matrix.png")

# ==============================================================================
# 6. Plot 2: Clean 2D t-SNE Projection
# ==============================================================================
fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
df["model_certainty"] = 1.0 - df["normalized_entropy"]

sns.scatterplot(
    data=df, x="tsne_x", y="tsne_y", hue="animal_category",
    size="model_certainty", sizes=(15, 220),
    alpha=0.65, palette="tab20", ax=ax, edgecolor="white", linewidth=0.3
)

for animal in AM2K_CATEGORIES:
    sub = df[df["animal_category"] == animal]
    if len(sub) > 5:
        cx, cy = sub["tsne_x"].median(), sub["tsne_y"].median()
        txt = ax.text(
            cx, cy, animal.upper(),
            fontsize=9.5, fontweight="black", color="#111111",
            ha="center", va="center", zorder=10
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3.5, foreground="white")])

top_traps = (
    df[df["top1_conf"] > 0.80]
    .sort_values("top1_conf", ascending=False)
    .groupby("animal_category", as_index=False)
    .first()
    .drop_duplicates(subset=["top1_pokemon"])
    .head(12)
)

center_x, center_y = df["tsne_x"].mean(), df["tsne_y"].mean()
span_x = df["tsne_x"].max() - df["tsne_x"].min()
span_y = df["tsne_y"].max() - df["tsne_y"].min()

angles = np.arctan2(top_traps["tsne_y"] - center_y, top_traps["tsne_x"] - center_x)
top_traps = top_traps.assign(angle=angles).sort_values("angle")

for idx, (_, row) in enumerate(top_traps.iterrows()):
    px, py = row["tsne_x"], row["tsne_y"]
    radial_dist = 0.12 if idx % 2 == 0 else 0.18
    dx = np.cos(row["angle"]) * span_x * radial_dist
    dy = np.sin(row["angle"]) * span_y * radial_dist

    ax.annotate(
        f"{row['top1_pokemon']}\n({row['animal_category']} -> {row['top1_conf']*100:.0f}%)",
        xy=(px, py),
        xytext=(px + dx, py + dy),
        fontsize=8, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D", lw=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#333333", lw=1.2),
        zorder=15
    )

ax.set_title(
    f"2D Visual Distance Map of {len(df):,} Mask Silhouettes (AM-2k)\n"
    "(Text Halo = Animal Cluster Center | Yellow Callout = Top High-Confidence Pokémon Attractor)",
    fontsize=13, fontweight="bold"
)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8.5, title="Animal & Certainty")
fig.tight_layout()
safe_savefig(fig, "phase1_mask_distance_tsne_projection.png")

# ==============================================================================
# 7. Plot 3: Bipartite Network Graph
# ==============================================================================
G_bip = nx.Graph()

edge_stats = (
    df.groupby(["animal_category", "top1_pokemon"])
    .agg(
        count=("top1_conf", "count"),
        mean_conf=("top1_conf", "mean"),
        mean_cert=("model_certainty", "mean")
    )
    .reset_index()
)
animal_counts = df["animal_category"].value_counts().to_dict()
edge_stats["pct_of_animal"] = edge_stats.apply(
    lambda r: (r["count"] / max(animal_counts.get(r["animal_category"], 1), 1)) * 100.0, axis=1
)

strong_edges = edge_stats[(edge_stats["pct_of_animal"] >= 8.0) & (edge_stats["count"] >= 3)].copy()

for _, row in strong_edges.iterrows():
    a_node = f"[A] {row['animal_category']}"
    p_node = f"[P] {row['top1_pokemon']}"
    G_bip.add_node(a_node, node_type="animal", size=450)
    poke_total = df[df["top1_pokemon"] == row["top1_pokemon"]].shape[0]
    G_bip.add_node(p_node, node_type="pokemon", size=300 + poke_total * 4)
    G_bip.add_edge(
        a_node, p_node,
        weight=row["pct_of_animal"] / 10.0,
        conf=row["mean_conf"],
        count=row["count"]
    )

fig, ax = plt.subplots(figsize=(15, 11), dpi=150)
pos = nx.spring_layout(G_bip, k=0.85, iterations=120, seed=42, weight="weight")

animal_nodes = [n for n, d in G_bip.nodes(data=True) if d["node_type"] == "animal"]
pokemon_nodes = [n for n, d in G_bip.nodes(data=True) if d["node_type"] == "pokemon"]

nx.draw_networkx_nodes(
    G_bip, pos, nodelist=animal_nodes, node_color="#64B5F6", node_shape="o",
    node_size=[G_bip.nodes[n]["size"] for n in animal_nodes], edgecolors="black", linewidths=1.2, ax=ax, label="Real Animal (AM-2k)"
)
nx.draw_networkx_nodes(
    G_bip, pos, nodelist=pokemon_nodes, node_color="#FF8A65", node_shape="D",
    node_size=[G_bip.nodes[n]["size"] for n in pokemon_nodes], edgecolors="black", linewidths=1.2, ax=ax, label="Pokémon Attractor"
)

edges = G_bip.edges(data=True)
widths = [d["weight"] * 1.8 for _, _, d in edges]
confs = [d["conf"] for _, _, d in edges]

edge_collection = nx.draw_networkx_edges(
    G_bip, pos, width=widths, edge_color=confs, edge_cmap=plt.cm.magma_r,
    edge_vmin=0.2, edge_vmax=0.9, alpha=0.85, ax=ax
)
cbar = plt.colorbar(edge_collection, ax=ax, shrink=0.7)
cbar.set_label("Mean Top-1 Confidence (Dark Purple = High Confidence | Light Orange = Low)", fontsize=10)

for node, (x, y) in pos.items():
    t = ax.text(x, y, node, fontsize=8.5, fontweight="bold", ha="center", va="center")
    t.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground="white")])

ax.set_title(
    "Mask Silhouette: Animal-to-Pokémon Bipartite Attractor Network\n"
    "(Blue Circle [A] = Real Animal | Orange Diamond [P] = Pokémon Sink | Edge Width = % Routed | Dark Edge = High Confidence)",
    fontsize=13, fontweight="bold"
)
ax.axis("off")
ax.legend(loc="upper left", fontsize=10)
fig.tight_layout()
safe_savefig(fig, "phase1_mask_bipartite_attractor_network.png")

# ==============================================================================
# 8. Plot 4: Pokémon-to-Pokémon Co-Confusion Network
# ==============================================================================
G_poke = nx.Graph()

if "top2_pokemon" in df.columns:
    valid_pairs = df[df["top1_conf"] > 0.25].copy()
    pair_records = []
    for _, r in valid_pairs.iterrows():
        p1, p2 = sorted([str(r["top1_pokemon"]), str(r["top2_pokemon"])])
        if p1 != p2:
            pair_records.append((p1, p2, r["top1_conf"], r["animal_category"]))
    pair_df = pd.DataFrame(pair_records, columns=["p1", "p2", "conf", "animal"])
else:
    pair_records = []
    top1_list = df["top1_pokemon"].tolist()
    conf_list = df["top1_conf"].tolist()
    anim_list = df["animal_category"].tolist()
    dist_copy = dist_matrix.copy()
    np.fill_diagonal(dist_copy, np.inf)
    nn_indices = np.argsort(dist_copy, axis=1)[:, :3]
    dist_thresh = np.percentile(dist_copy[dist_copy < np.inf], 5)
    for i in range(len(df)):
        if conf_list[i] < 0.35:
            continue
        for j in nn_indices[i]:
            if dist_copy[i, j] <= dist_thresh and top1_list[i] != top1_list[j]:
                p1, p2 = sorted([str(top1_list[i]), str(top1_list[j])])
                pair_records.append((p1, p2, (conf_list[i] + conf_list[j]) / 2.0, anim_list[i]))
    pair_df = pd.DataFrame(pair_records, columns=["p1", "p2", "conf", "animal"])

co_edges = (
    pair_df.groupby(["p1", "p2"])
    .agg(
        shared_count=("conf", "count"),
        mean_conf=("conf", "mean"),
        dominant_animal=("animal", lambda x: x.value_counts().index[0])
    )
    .reset_index()
    .sort_values("shared_count", ascending=False)
    .head(35)
)

for _, row in co_edges.iterrows():
    G_poke.add_edge(
        row["p1"], row["p2"],
        weight=row["shared_count"],
        conf=row["mean_conf"],
        animal=row["dominant_animal"]
    )

fig, ax = plt.subplots(figsize=(13, 10), dpi=150)
pos_p = nx.kamada_kawai_layout(G_poke)

node_sizes = [350 + df[df["top1_pokemon"] == n].shape[0] * 6 for n in G_poke.nodes()]
nx.draw_networkx_nodes(
    G_poke, pos_p, node_size=node_sizes, node_color="#AED581",
    edgecolors="#33691E", linewidths=1.5, ax=ax
)

p_edges = G_poke.edges(data=True)
max_count = max([d["weight"] for _, _, d in p_edges], default=1)
p_widths = [1.0 + (d["weight"] / max_count) * 6.0 for _, _, d in p_edges]
p_confs = [d["conf"] for _, _, d in p_edges]

ec = nx.draw_networkx_edges(
    G_poke, pos_p, width=p_widths, edge_color=p_confs,
    edge_cmap=plt.cm.magma_r, edge_vmin=0.3, edge_vmax=0.9, alpha=0.85, ax=ax
)
cbar2 = plt.colorbar(ec, ax=ax, shrink=0.7)
cbar2.set_label("Mean Confidence of Connected Images (Dark Purple = High Confidence)", fontsize=10)

for node, (x, y) in pos_p.items():
    t = ax.text(x, y, node, fontsize=9, fontweight="bold", ha="center", va="center")
    t.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground="white")])

if len(p_edges) > 0:
    weight_thresh = np.percentile([x[2]["weight"] for x in p_edges], 50)
    edge_labels = {(u, v): d["animal"] for u, v, d in p_edges if d["weight"] >= weight_thresh}
    nx.draw_networkx_edge_labels(G_poke, pos_p, edge_labels=edge_labels, font_size=7.5, font_color="#B71C1C", ax=ax)

ax.set_title(
    "Mask Silhouette: Pokémon-to-Pokémon OOD Co-Confusion Graph\n"
    "(Edge Width = Shared Animal Images / Visual Neighbors | Red Edge Label = Animal Bridging the Two Pokémon)",
    fontsize=12.5, fontweight="bold"
)
ax.axis("off")
fig.tight_layout()
safe_savefig(fig, "phase1_mask_pokemon_coconfusion_network.png")