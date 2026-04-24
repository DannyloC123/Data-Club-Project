# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import pickle

import nltk
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from transformers import AutoTokenizer, AutoModel, pipeline
# import hdbscan


# ── GLOBAL VARIABLES & CONFIGURATION ─────────────────────────────────────────

# Path to the input data file
file_path = 'Review_Data.xlsx'

# When True, limits processing to a smaller sample for faster iteration during development
DEV_MODE = True

# Number of reviews to embed/cluster at once; larger = faster but more memory
batch_size = 64

# Words to strip from TF-IDF keywords that are brand names, product categories,
# or other terms that don't describe a product attribute
custom_stopwords = {
    "cascade", "tide", "gain", "downy", "dawn",
    "febreze", "swiffer", "bounce", "platinum",
    "magic",
    "dishwasher", "dishes", "laundry", "detergent",
    "pods", "clothes", "fabric", "softener",
    "soap", "cleaner", "washing",
    "food", "item", "product", "brand", "pack", "bottle"
}

# Generic sentiment words we don't want surfaced as cluster keywords
bad_words = {
    "good", "great", "best", "nice", "bad", "poor",
    "excellent", "perfect", "amazing",
    "love", "loves", "liked", "like",
    "works", "working", "use", "used"
}


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """
    Converts per-token embeddings into a single sentence embedding by averaging
    all token vectors, weighted so that padding tokens don't contribute.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def dedupe_keywords(keywords):
    """
    Removes near-duplicate keywords (e.g. 'clean' and 'cleans') by comparing
    word stems. Keeps the first occurrence and drops later variants.
    """
    seen = set()
    result = []
    for w in keywords:
        root = w.rstrip('s')  # simple normalization
        if root not in seen:
            seen.add(root)
            result.append(w)
    return result


def keep_descriptive_words(words):
    """
    Filters a word list down to adjectives (JJ*) and nouns (NN*) using POS tagging,
    then removes any generic sentiment words that don't describe product attributes.
    """
    tagged = nltk.pos_tag(words)

    return [
        word for word, tag in tagged
        if (
            (tag.startswith("JJ") or tag.startswith("NN"))
            and word.lower() not in bad_words
        )
    ]


# ── SETUP: DATA & MODELS ──────────────────────────────────────────────────────

# Download NLTK data needed for POS tagging used in keyword extraction
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found.\nChecked path: {file_path}")

# Pre-trained DistilBERT model fine-tuned on SST-2 for positive/negative classification
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

pg_dataframe = pd.read_excel(file_path)


# ── SETTINGS ──────────────────────────────────────────────────────────────────

# In dev mode we cap the sample size to avoid waiting on the full dataset
if DEV_MODE:
    num_samples = 3000
else:
    num_samples = len(pg_dataframe)

#pg_subset = pg_dataframe.sample(n=num_samples, random_state=42).copy()
#num_samples = len(pg_dataframe)  # change this to control how many descriptions you use
pg_subset = pg_dataframe.copy()


# ── GROUND-TRUTH LABELS ───────────────────────────────────────────────────────

# Ratings ≥ 4 are "positive" (1), ratings ≤ 2 are "negative" (0).
# Rating 3 is ambiguous, so we drop those rows entirely.
pg_subset["label"] = pg_subset["review_rating"].apply(
    lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
)

pg_subset = pg_subset.dropna(subset=["label"])


# ── SENTIMENT MODEL EVALUATION ────────────────────────────────────────────────

# Run the sentiment model on a random 1 000-review sample so evaluation is fast
sentiment_sample = pg_subset.sample(n=1000, random_state=42)

clean_texts = [
    str(x) for x in sentiment_sample["review_text"].fillna("")
]

preds = sentiment_model(
    clean_texts,
    batch_size=32,
    truncation=True,
    max_length=512
)

# Map the model's string label back to our 0/1 scheme
sentiment_sample["predicted_label"] = [
    1 if p["label"] == "POSITIVE" else 0
    for p in preds
]

acc = accuracy_score(
    sentiment_sample["label"],
    sentiment_sample["predicted_label"]
)

print(f"\n🎯 Sentiment Accuracy: {acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(
    sentiment_sample["label"],
    sentiment_sample["predicted_label"]
))


# ── SENTENCE EMBEDDINGS ───────────────────────────────────────────────────────

# We embed review_text so that semantically similar reviews end up close
# together in vector space, enabling meaningful clustering.
sentences = pg_subset["review_text"].fillna("").astype(str).tolist()

# all-MiniLM-L6-v2 is a lightweight sentence-transformer that balances
# speed and quality for sentence-level semantic similarity tasks.
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model.eval()

all_embeddings = []

# Reuse saved embeddings to avoid re-running the expensive forward pass each time
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

    # Concatenate all batch tensors into one matrix, then cache to disk
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(sentence_embeddings, "embeddings.pt")

    print("Embeddings saved.")


# ── CLUSTERING ────────────────────────────────────────────────────────────────

# K-Means groups reviews into 10 clusters based on embedding similarity.
# Each cluster ideally represents a coherent theme or product attribute.
kmeans = KMeans(n_clusters=10, random_state=42)
pg_subset["cluster"] = kmeans.fit_predict(sentence_embeddings.cpu().numpy())

# Silhouette score measures how well-separated the clusters are (-1 to 1;
# higher is better, >0.2 is generally considered meaningful).
sil_score = silhouette_score(
    sentence_embeddings.cpu().numpy(),
    pg_subset["cluster"]
)

print(f"\n📐 Silhouette Score: {sil_score:.4f}")


# ── CLUSTER → SENTIMENT ACCURACY ─────────────────────────────────────────────

# Assign each cluster the majority sentiment label from its members,
# then measure how well that majority label predicts individual reviews.
cluster_to_label = (
    pg_subset.groupby("cluster")["label"]
    .mean()
    .round()
)

pred_from_cluster = pg_subset["cluster"].map(cluster_to_label)

cluster_acc = accuracy_score(pg_subset["label"], pred_from_cluster)

print(f"\n📊 Cluster-based Accuracy: {cluster_acc:.4f}")


# ── CLUSTER SUMMARY & RANKING ─────────────────────────────────────────────────

cluster_stats = pg_subset.groupby("cluster")["review_rating"].agg(["mean", "count"])

# Rank clusters from most positive to least, using positive_rate as the primary
# sort key and avg_rating as a tiebreaker.
cluster_summary = pg_subset.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    positive_rate=("label", "mean"),
    size=("cluster", "count")
).reset_index()

cluster_summary = cluster_summary.sort_values(
    by=["positive_rate", "avg_rating"],
    ascending=False
)

cluster_summary["rank"] = range(1, len(cluster_summary) + 1)


# ── TF-IDF KEYWORD EXTRACTION ─────────────────────────────────────────────────

# TF-IDF surfaces words that are common within a cluster but rare across
# all clusters — these become the descriptive keywords / theme labels.
all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

tfidf = TfidfVectorizer(
    stop_words=all_stopwords,
    max_features=3000,
    min_df=10,
    max_df=0.8
)

# Reuse cached TF-IDF to avoid refitting on every run
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

# Pre-compute the mean TF-IDF vector for each cluster so we only do this once
cluster_means = {}
cluster_ids = pg_subset["cluster"].values

for c in cluster_summary["cluster"]:
    mask = (cluster_ids == c)
    cluster_means[c] = tfidf_matrix[mask].mean(axis=0).A1


# clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
# pg_subset["cluster"] = clusterer.fit_predict(sentence_embeddings.cpu().numpy())


# ── CLUSTER LABEL GENERATION ──────────────────────────────────────────────────

# For each cluster: grab top TF-IDF terms → filter noise → keep adjectives/nouns
# → dedupe → join into a short human-readable label (e.g. "clean / fresh / scent").
cluster_table = cluster_summary.copy()
labels = {}

for c in cluster_summary["cluster"]:
    mean_tfidf = cluster_means[c]
    top_indices = mean_tfidf.argsort()[-8:][::-1]

    keywords = [terms[i] for i in top_indices]

    filtered_keywords = [
        w for w in keywords
        if (
            w not in custom_stopwords
            and w.lower() not in bad_words
            and len(w) > 3
        )
    ]

    descriptive = keep_descriptive_words(filtered_keywords)

    deduped = dedupe_keywords(descriptive)

    # If filtering removed everything, fall back to the raw filtered list
    if len(deduped) == 0:
        deduped = filtered_keywords[:3]

    if len(deduped) >= 2:
        labels[c] = " / ".join(deduped[:3])
    else:
        fallback = dedupe_keywords([
            w for w in filtered_keywords
            if w not in {
                "good", "great", "best", "nice", "bad", "poor",
                "excellent", "perfect", "amazing",
                "love", "loves", "liked", "like",
                "works", "working", "use", "used"
            }
        ])

        if len(fallback) >= 2:
            labels[c] = " / ".join(fallback[:3])
        elif len(filtered_keywords) >= 2:
            labels[c] = " / ".join(filtered_keywords[:2])  # last resort
        else:
            labels[c] = filtered_keywords[0] if filtered_keywords else "other"


# Attach human-readable labels back to the summary table, then sort by rank
cluster_summary["label"] = cluster_summary["cluster"].map(labels)

cluster_table = cluster_table.sort_values(by="rank")

print("\n📊 CLUSTER SUMMARY TABLE:\n")
print(cluster_table)


# ── NEGATIVE CLUSTER ANALYSIS ─────────────────────────────────────────────────

# Separate positive and negative reviews so we can inspect which clusters
# are driving the most negative feedback.
positive_df = pg_subset[pg_subset["label"] == 1]
negative_df = pg_subset[pg_subset["label"] == 0]

neg_cluster_summary = negative_df.groupby("cluster").agg(
    avg_rating=("review_rating", "mean"),
    size=("cluster", "count")
).reset_index()

neg_cluster_summary = neg_cluster_summary.sort_values(
    by="size", ascending=False
)

neg_cluster_summary["label"] = neg_cluster_summary["cluster"].map(labels)

print("\n❌ NEGATIVE CLUSTERS WITH LABELS:\n")
print(neg_cluster_summary[["cluster", "label", "avg_rating", "size"]].head(10))


# ── SAVE OUTPUT ───────────────────────────────────────────────────────────────

# Write the final dataframe — now enriched with cluster ID and label — to CSV
# so results can be explored in Excel or passed to downstream tools.
pg_subset["cluster_label"] = pg_subset["cluster"].map(labels)
output_path = 'Subset_with_embeddings.csv'
pg_subset.to_csv(output_path, index=False)

print("Saved to:", output_path)