# 🧼 Product Review Analysis with NLP

This project analyzes product reviews using Natural Language Processing (NLP) to extract meaningful insights about customer sentiment and key product attributes.

It combines:
- Sentiment classification (supervised)
- Text embeddings (BERT)
- Clustering (unsupervised)
- Theme extraction (TF-IDF + keyword filtering)

---

# 🚀 Features

### 1. Sentiment Analysis
- Uses a pretrained model from Hugging Face
- Compares predicted sentiment vs review ratings
- Outputs accuracy + classification metrics

### 2. Clustering
- Converts reviews into embeddings using BERT
- Groups similar reviews using KMeans
- Evaluates clustering quality (silhouette score)

### 3. Theme Extraction
- Uses TF-IDF to extract keywords
- Filters for descriptive words (adjectives)
- Produces human-readable labels like:
  - `clean / dish`
  - `fresh / scent`
  - `price / value`

### 4. Theme Grouping
- Groups similar labels into higher-level themes:
  - Cleaning Effectiveness
  - Scent / Smell
  - Price / Value
  - Ease of Use
  - Product Experience

---

# 📊 Outputs

### Model Metrics
- Sentiment Accuracy
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Silhouette Score (clustering quality)
- Cluster-based Accuracy (alignment with sentiment)

### Data Outputs
- `Subset_with_embeddings.csv` (processed dataset with labels)

### Visualizations (optional)
- Attribute ranking charts
- Theme performance charts
- Importance vs satisfaction scatter plots

main.py:

<img width="609" height="292" alt="image" src="https://github.com/user-attachments/assets/742a1cc2-0c4e-414f-9cbe-04d563c91f6d" />



graphs.py (so far):

<img width="719" height="598" alt="image" src="https://github.com/user-attachments/assets/b79d92db-198b-4683-9bc3-919f1fe1223c" />


---

# 🐳 Docker Setup

## Build the image

```bash
sudo docker build -t review-analysis .
```

## Run 

```bash
sudo docker run -v $(pwd):/app review-analysis

sudo docker run -v $(pwd):/app review-analysis python graphs.py
```
