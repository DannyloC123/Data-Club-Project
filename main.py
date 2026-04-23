from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import os
import nltk
# import hdbscan


# -----------------------------
# 🔹 LOAD DATA
# -----------------------------

file_path = 'Review_Data.xlsx'
DEV_MODE = True

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found.\nChecked path: {file_path}")

pg_dataframe = pd.read_excel(file_path)

# -----------------------------
# 🔹 SETTINGS
# -----------------------------

if DEV_MODE:
    num_samples = 3000
else:
    num_samples = len(pg_dataframe)
#pg_subset = pg_dataframe.sample(n=num_samples, random_state=42).copy()
#num_samples = len(pg_dataframe)  # change this to control how many descriptions you use
pg_subset = pg_dataframe.copy()
batch_size = 64


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

if os.path.exists("embeddings.pt"):
    print("Loading cached embeddings...")
    sentence_embeddings = torch.load("embeddings.pt")

else:
    print("Generating embeddings...")

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

        print(f"Processed {i + len(batch)} / {len(sentences)}", flush=True)

    # -----------------------------
    # 🔹 COMBINE + SAVE (ONLY HERE)
    # -----------------------------
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(sentence_embeddings, "embeddings.pt")

    print("Embeddings saved.")




# -----------------------------
# 🔹 OPTIONAL: CLUSTERING
# -----------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
pg_subset["cluster"] = kmeans.fit_predict(sentence_embeddings.cpu().numpy())

# -----------------------------
# 🔹 CLUSTER → RATING ANALYSIS
# -----------------------------
#print("\nCluster rating analysis:")

cluster_stats = pg_subset.groupby("cluster")["review_rating"].agg(["mean", "count"])
#print(cluster_stats)



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

#print("\nCluster Ranking:")
#print(cluster_summary)

# -----------------------------
# 🔹 CLUSTER → AUTO KEYWORDS (IMPROVED)
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

custom_stopwords = {
    # 🔹 BRAND / PRODUCT NAMES
    "cascade", "tide", "gain", "downy", "dawn",
    "febreze", "swiffer", "bounce", "platinum",
    "magic",

    # 🔹 GENERIC PRODUCT CATEGORY WORDS (non-attributes)
    "dishwasher", "dishes", "laundry", "detergent",
    "pods", "clothes", "fabric", "softener",
    "soap", "cleaner", "washing",

    # 🔹 REDUNDANT / NON-ATTRIBUTE TERMS
    "food", "item", "product", "brand", "pack", "bottle"
}

all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

tfidf = TfidfVectorizer(
    stop_words=all_stopwords,
    max_features=3000,
    min_df=10,
    max_df=0.8
)

# -----------------------------
# 🔹 LOAD / GENERATE TF-IDF
# -----------------------------
import pickle

if os.path.exists("tfidf.pkl") and os.path.exists("tfidf_matrix.pkl"):
    print("Loading cached TF-IDF...")

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

else:
    print("Generating TF-IDF...")

    tfidf_matrix = tfidf.fit_transform(
        pg_subset["review_text"].fillna("").astype(str)
    )

    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

terms = tfidf.get_feature_names_out()

# -----------------------------
# 🔹 PRECOMPUTE CLUSTER TF-IDF MEANS
# -----------------------------
cluster_means = {}

cluster_ids = pg_subset["cluster"].values

for c in cluster_summary["cluster"]:
    mask = (cluster_ids == c)
    cluster_means[c] = tfidf_matrix[mask].mean(axis=0).A1


#print("\n🔍 CLUSTER THEMES:\n")


# clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
# pg_subset["cluster"] = clusterer.fit_predict(sentence_embeddings.cpu().numpy())


# -----------------------------
# 🔹 BUILD CLUSTER SUMMARY TABLE
# -----------------------------
cluster_table = cluster_summary.copy()


# -----------------------------
# 🔹 ADD LABELS (FIXED)
# -----------------------------
labels = {}

# -----------------------------
# 🔹 DEDUPE KEYWORDS
# -----------------------------
def dedupe_keywords(keywords):
    seen = set()
    result = []
    for w in keywords:
        root = w.rstrip('s')  # simple normalization
        if root not in seen:
            seen.add(root)
            result.append(w)
    return result

def keep_descriptive_words(words):
    tagged = nltk.pos_tag(words)
    return [
        word for word, tag in tagged
        if tag.startswith("JJ")  # adjectives only
    ]

for c in cluster_summary["cluster"]:
    mean_tfidf = cluster_means[c]
    top_indices = mean_tfidf.argsort()[-8:][::-1]

    keywords = [terms[i] for i in top_indices]

    # filter junk words
    filtered_keywords = [
        w for w in keywords
        if w not in custom_stopwords and len(w) > 3
    ]

    # 🔥 KEEP ONLY ATTRIBUTES (adjectives)
    descriptive = keep_descriptive_words(filtered_keywords)

    # dedupe after filtering
    deduped = dedupe_keywords(descriptive)

    # fallback if empty
    if len(deduped) == 0:
        deduped = filtered_keywords[:3]

    if len(deduped) >= 2:
        labels[c] = " / ".join(deduped[:3])
    else:
        # fallback: mix in other filtered words
        fallback = dedupe_keywords(filtered_keywords)
        labels[c] = " / ".join(fallback[:3])


# map labels to table
cluster_table["label"] = cluster_table["cluster"].map(labels)

# Optional: sort by rank or size
cluster_table = cluster_table.sort_values(by="rank")

print("\n📊 CLUSTER SUMMARY TABLE:\n")
print(cluster_table)


# -----------------------------
# 🔹 VIEW RESULTS
# -----------------------------
#print("\nCluster breakdown:")
#print(pg_subset.groupby("cluster").size())
'''
for c in cluster_summary["cluster"]:
    print(f"\nCluster {c} examples:")
    print(pg_subset[pg_subset["cluster"] == c]["review_text"].head(3))
'''


# -----------------------------
# 🔹 SAVE FILE
# -----------------------------
pg_subset["cluster_label"] = pg_subset["cluster"].map(labels)
output_path = 'Subset_with_embeddings.csv'
pg_subset.to_csv(output_path, index=False)

print("Saved to:", output_path)