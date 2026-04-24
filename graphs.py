# ── IMPORTS ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# ── GLOBAL VARIABLES & CONFIGURATION ─────────────────────────────────────────

# How many semantic themes to group the cluster labels into.
# Tune this between 5–8 depending on how broad or granular you want the themes.
n_themes = 5

# Manual names assigned to each theme after inspecting the printed groupings.
# Update these after running the script and reviewing the "THEMES" output.
theme_names = {
    0: "Cleaning Effectiveness",
    1: "Scent / Smell",
    2: "Price / Value",
    3: "Ease of Use / Convenience",
    4: "Product Experience"
}

# Words that add no descriptive value to a cluster label and should be stripped out
bad_words = {
    "good", "great", "best", "product", "review", "collected",
    "promotion", "time", "does", "gets", "exactly", "work"
}


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def clean_label(label):
    """
    Cleans a raw cluster label string (e.g. "clean / great / soap") by:
    - removing generic/uninformative words
    - deduplicating
    - dropping words shorter than 4 characters
    Returns a concise "word1 / word2" string, a single word, or "" if nothing useful remains.
    """
    words = label.split(" / ")

    filtered = [w for w in words if w not in bad_words]
    filtered = list(dict.fromkeys(filtered))
    filtered = [w for w in filtered if len(w) > 3]

    if len(filtered) >= 2:
        return f"{filtered[0]} / {filtered[1]}"
    elif len(filtered) == 1:
        return filtered[0]
    else:
        return ""


# ── LOAD & PREPARE DATA ───────────────────────────────────────────────────────

# Load the enriched CSV produced by the embedding/clustering script
df = pd.read_csv("Subset_with_embeddings.csv")

# Isolate negative reviews (label == 0) for complaint-focused analysis later
negative_df = df[df["label"] == 0]


# ── CLUSTER SUMMARIES ─────────────────────────────────────────────────────────

# Aggregate stats for negative clusters — used at the end for complaint theme graph
neg_cluster_summary = negative_df.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    size=("cluster", "count")
).reset_index()

neg_cluster_summary = neg_cluster_summary.sort_values(
    by="size", ascending=False
)

# Rebuild the full cluster summary table from scratch so this script is self-contained
cluster_table = df.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    positive_rate=("label", "mean"),
    size=("cluster", "count")
).reset_index()


# ── LABEL CLEANING ────────────────────────────────────────────────────────────

# Pull the first cluster_label value per cluster (they're all the same within a cluster)
# then run each through clean_label to strip noise
raw_labels = df.groupby("cluster")["cluster_label"].first().to_dict()

labels = {
    k: clean_label(v)
    for k, v in raw_labels.items()
}

# Attach cleaned labels to both tables
neg_cluster_summary["label"] = neg_cluster_summary["cluster"].map(labels)
cluster_table["label"] = cluster_table["cluster"].map(labels)

# Drop any cluster whose label was completely filtered out — it has nothing useful to say
cluster_table = cluster_table[cluster_table["label"] != ""]


# ── SORT & RANK CLUSTERS ──────────────────────────────────────────────────────

# Rank clusters by average rating so the best-performing attributes appear first
cluster_table = cluster_table.sort_values(
    by="avg_rating", ascending=False
).reset_index(drop=True)

cluster_table["rank"] = cluster_table.index + 1


# ── THEME DETECTION ───────────────────────────────────────────────────────────

# Encode the cleaned label strings into embeddings so semantically similar labels
# (e.g. "clean / fresh" and "clean / rinse") get grouped into the same theme
model = SentenceTransformer('all-MiniLM-L6-v2')

labels_list = cluster_table["label"].astype(str).tolist()
label_embeddings = model.encode(labels_list)

# K-Means on the label embeddings groups them into n_themes broad themes
kmeans = KMeans(n_clusters=n_themes, random_state=42)
cluster_table["theme"] = kmeans.fit_predict(label_embeddings)

# Print the groupings so you can decide what to name each theme_id in theme_names above
print("\n📊 THEMES:\n")
for t in range(n_themes):
    print(f"Theme {t}:")
    print(cluster_table[cluster_table["theme"] == t]["label"].tolist())
    print()

cluster_table["theme_name"] = cluster_table["theme"].map(theme_names)


# ── THEME-LEVEL SUMMARY ───────────────────────────────────────────────────────

# Roll up cluster stats to the theme level for a higher-level view of performance
theme_summary = cluster_table.groupby("theme_name").agg(
    avg_rating=("avg_rating", "mean"),
    total_size=("size", "sum")
).sort_values(by="avg_rating", ascending=False)

print("\n📊 THEME SUMMARY:\n")
print(theme_summary)


# ── GRAPHS ────────────────────────────────────────────────────────────────────

# Graph 1: Which specific attributes correlate with the highest ratings?
top = cluster_table.head(15)

plt.figure()
plt.barh(top["label"], top["avg_rating"])
plt.xlabel("Average Rating")
plt.title("Top Product Attributes Driving High Ratings")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graph_top_attributes.png")
plt.close()

# Graph 2: Which attributes appear most frequently — i.e. what do customers talk about most?
plt.figure()
plt.barh(top["label"], top["size"])
plt.xlabel("Number of Reviews")
plt.title("Most Common Product Attributes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graph_attribute_frequency.png")
plt.close()

# Graph 3: Importance vs Satisfaction scatter.
# Points in the top-right quadrant (high frequency AND high rating) are highlighted
# in red — these are the key drivers worth prioritizing.
plt.figure(figsize=(10, 6))

x = cluster_table["size"]
y = cluster_table["avg_rating"]

x_mean = x.mean()
y_mean = y.mean()

colors = []
for xi, yi in zip(x, y):
    if xi > x_mean and yi > y_mean:
        colors.append("red")
    else:
        colors.append("gray")

plt.scatter(x, y, c=colors)

for i, txt in enumerate(cluster_table["label"]):
    plt.annotate(
        txt,
        (x.iloc[i], y.iloc[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("Number of Reviews (Importance)")
plt.ylabel("Average Rating (Satisfaction)")
plt.title("Attribute Importance vs Satisfaction")

plt.axhline(y=y_mean, linestyle="--")
plt.axvline(x=x_mean, linestyle="--")

plt.tight_layout()
plt.savefig("graph_importance_vs_satisfaction.png")
plt.close()

# Graph 4: Average rating rolled up to theme level — shows which broad categories
# are performing well vs. dragging down overall satisfaction
plt.figure()
plt.barh(theme_summary.index, theme_summary["avg_rating"])
plt.xlabel("Average Rating")
plt.title("Performance by Product Theme")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graph_theme_performance.png")
plt.close()


# ── NEGATIVE THEME ANALYSIS ───────────────────────────────────────────────────

# Map each negative review's cluster to its theme name, then count how many
# negative reviews fall into each theme to find the biggest complaint areas
cluster_to_theme = cluster_table.set_index("cluster")["theme_name"].to_dict()

negative_df["theme_name"] = negative_df["cluster"].map(cluster_to_theme)

neg_theme_summary = negative_df.groupby("theme_name").agg(
    size=("cluster", "count"),
    avg_rating=("review_rating", "mean")
).reset_index()

neg_theme_summary = neg_theme_summary.sort_values(
    by="size", ascending=False
)

# Graph 5: Which themes generate the most negative reviews?
plt.figure()
top_neg = neg_theme_summary.head(5)
plt.barh(top_neg["theme_name"], top_neg["size"])
plt.xlabel("Number of Negative Reviews")
plt.title("Top Customer Complaint Themes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graph_top_complaints.png")
plt.close()