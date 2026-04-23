from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import os

# -----------------------------
# 🔹 SETTINGS
# -----------------------------
num_samples = 56265   # change this to control how many descriptions you use
batch_size = 32

# -----------------------------
# 🔹 LOAD DATA
# -----------------------------

file_path = 'Review_Data.xlsx'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found.\nChecked path: {file_path}")

pg_dataframe = pd.read_excel(file_path)


# -----------------------------
# 🔹 SELECT SUBSET (IMPORTANT FIX)
# -----------------------------
pg_subset = pg_dataframe.sample(n=num_samples, random_state=42).copy()

# -----------------------------
# 🔹 DEFINE HIGH VS LOW REVIEWS (GROUND TRUTH)
# -----------------------------
pg_subset["label"] = pg_subset["review_rating"].apply(
    lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
)

# Remove neutral reviews (rating = 3)
pg_subset = pg_subset.dropna(subset=["label"])


# Use product descriptions (you can switch to review_text if needed)
sentences = pg_subset["review_text"].fillna("").astype(str).tolist()


# -----------------------------
# 🔹 MEAN POOLING FUNCTION
# -----------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# -----------------------------
# 🔹 LOAD MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model.eval()


# -----------------------------
# 🔹 GENERATE EMBEDDINGS (BATCHED)
# -----------------------------
all_embeddings = []

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]

    encoded_input = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

    all_embeddings.append(batch_embeddings)

    print(f"Processed {i + len(batch)} / {len(sentences)}")


# -----------------------------
# 🔹 COMBINE RESULTS
# -----------------------------
sentence_embeddings = torch.cat(all_embeddings, dim=0)

print("Final embedding shape:", sentence_embeddings.shape)


# -----------------------------
# 🔹 STORE RESULTS (FIXED)
# -----------------------------
pg_subset["embedding"] = sentence_embeddings.tolist()


# -----------------------------
# 🔹 OPTIONAL: CLUSTERING
# -----------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
pg_subset["cluster"] = kmeans.fit_predict(sentence_embeddings.cpu().numpy())

# -----------------------------
# 🔹 CLUSTER → RATING ANALYSIS
# -----------------------------
print("\nCluster rating analysis:")

cluster_stats = pg_subset.groupby("cluster")["review_rating"].agg(["mean", "count"])
print(cluster_stats)


# -----------------------------
# 🔹 CLUSTER → POSITIVE RATE
# -----------------------------
cluster_label_stats = pg_subset.groupby("cluster")["label"].mean()

print("\n% of positive reviews per cluster:")
print(cluster_label_stats)


# -----------------------------
# 🔹 CLUSTER → RANKING
# -----------------------------
cluster_summary = pg_subset.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    positive_rate=("label", "mean"),
    size=("cluster", "count")
).reset_index()

# Sort clusters (best → worst)
cluster_summary = cluster_summary.sort_values(
    by=["positive_rate", "avg_rating"],
    ascending=False
)

# Add rank column
cluster_summary["rank"] = range(1, len(cluster_summary) + 1)

print("\nCluster Ranking:")
print(cluster_summary)

# -----------------------------
# 🔹 CLUSTER → AUTO KEYWORDS
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(
    pg_subset["review_text"].fillna("").astype(str)
)
terms = tfidf.get_feature_names_out()

cluster_keywords = {}

for c in cluster_summary["cluster"]:
    cluster_indices = pg_subset["cluster"] == c
    cluster_tfidf = tfidf_matrix[cluster_indices.values]

    mean_tfidf = cluster_tfidf.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-5:][::-1]

    keywords = [terms[i] for i in top_indices]
    cluster_keywords[c] = keywords


# -----------------------------
# 🔹 VIEW RESULTS
# -----------------------------
print("\nCluster breakdown:")
print(pg_subset.groupby("cluster").size())

for c in range(3):
    print(f"\nCluster {c} examples:")
    print(pg_subset[pg_subset["cluster"] == c]["review_text"].head(3))


# -----------------------------
# 🔹 SAVE FILE
# -----------------------------
output_path = 'Subset_with_embeddings.csv'
pg_subset.to_csv(output_path, index=False)

print("Saved to:", output_path)