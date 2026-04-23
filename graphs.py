import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# -----------------------------
# 🔹 LOAD DATA
# -----------------------------
df = pd.read_csv("Subset_with_embeddings.csv")

# -----------------------------
# 🔹 REBUILD CLUSTER TABLE
# -----------------------------
cluster_table = df.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    positive_rate=("label", "mean"),
    size=("cluster", "count")
).reset_index()

# -----------------------------
# 🔹 ADD LABELS BACK
# -----------------------------
labels = df.groupby("cluster")["cluster_label"].first().to_dict()
cluster_table["label"] = cluster_table["cluster"].map(labels)

# -----------------------------
# 🔹 SORT + RANK
# -----------------------------
cluster_table = cluster_table.sort_values(
    by="avg_rating", ascending=False
).reset_index(drop=True)

cluster_table["rank"] = cluster_table.index + 1

# -----------------------------
# 🔹 EMBED LABELS (FOR THEMES)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

cluster_table = cluster_table.dropna(subset=["label"])
labels_list = cluster_table["label"].astype(str).tolist()
label_embeddings = model.encode(labels_list)

# -----------------------------
# 🔹 CLUSTER LABELS INTO THEMES
# -----------------------------
n_themes = 5  # adjust (5–8 works well)

kmeans = KMeans(n_clusters=n_themes, random_state=42)
cluster_table["theme"] = kmeans.fit_predict(label_embeddings)

# -----------------------------
# 🔹 PRINT GROUPED LABELS (FOR INTERPRETATION)
# -----------------------------
print("\n📊 THEMES:\n")
for t in range(n_themes):
    print(f"Theme {t}:")
    print(cluster_table[cluster_table["theme"] == t]["label"].tolist())
    print()

# -----------------------------
# 🔹 MANUAL THEME NAMES (EDIT THESE)
# -----------------------------
theme_names = {
    0: "Cleaning Effectiveness",
    1: "Scent / Smell",
    2: "Price / Value",
    3: "Ease of Use / Convenience",
    4: "Product Experience"
}

cluster_table["theme_name"] = cluster_table["theme"].map(theme_names)

# -----------------------------
# 🔹 FINAL THEME SUMMARY
# -----------------------------
theme_summary = cluster_table.groupby("theme_name").agg(
    avg_rating=("avg_rating", "mean"),
    total_size=("size", "sum")
).sort_values(by="avg_rating", ascending=False)

print("\n📊 THEME SUMMARY:\n")
print(theme_summary)

# -----------------------------
# 🔹 GRAPH 1: TOP ATTRIBUTES
# -----------------------------
top = cluster_table.head(15)

plt.figure()
plt.barh(top["label"], top["avg_rating"])
plt.xlabel("Average Rating")
plt.title("Top Product Attributes Driving High Ratings")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# 🔹 GRAPH 2: ATTRIBUTE FREQUENCY
# -----------------------------
plt.figure()
plt.barh(top["label"], top["size"])
plt.xlabel("Number of Reviews")
plt.title("Most Common Product Attributes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# 🔹 GRAPH 3: IMPORTANCE VS SATISFACTION
# -----------------------------
plt.figure()
plt.scatter(cluster_table["size"], cluster_table["avg_rating"])

for i, txt in enumerate(cluster_table["label"]):
    plt.annotate(
        txt,
        (cluster_table["size"].iloc[i],
         cluster_table["avg_rating"].iloc[i]),
        fontsize=8
    )

plt.xlabel("Number of Reviews (Importance)")
plt.ylabel("Average Rating (Satisfaction)")
plt.title("Attribute Importance vs Satisfaction")
plt.tight_layout()
plt.show()

# -----------------------------
# 🔹 GRAPH 4: THEME-LEVEL VIEW
# -----------------------------
plt.figure()
plt.barh(theme_summary.index, theme_summary["avg_rating"])
plt.xlabel("Average Rating")
plt.title("Performance by Product Theme")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()