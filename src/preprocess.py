import sys
import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import get_cluster_topic_names
from src.vector_store import InternalVectorDB
from src.config import config
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import re

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()


def aggressive_clean(text):

    # Remove email headers, subjects, and other non-essential parts
    text = re.sub(
        r"(?m)^(From|Subject|Lines|Organization|Reply-To|Nntp-Posting-Host):.*$",
        "",
        text,
    )
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)

    # Remove non-alphabetical noise and convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()

    # Lemmatization step
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words if len(w) > 3]

    # Final join and filter
    text = " ".join(lemmatized_words)
    return text if len(lemmatized_words) > 10 else None


def main():
    os.makedirs("data", exist_ok=True)

    # Load & Clean Data
    print("Loading 20 Newsgroups data...")
    newsgroups = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    print("Cleaning documents...")
    cleaned_docs = [aggressive_clean(doc) for doc in newsgroups.data]
    docs = [d for d in cleaned_docs if d is not None]

    # Embeddings (384-D)
    embedding_cache_path = "data/raw_embeddings.npy"
    if os.path.exists(embedding_cache_path):
        embeddings = np.load(embedding_cache_path)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(docs, show_progress_bar=True)
        np.save(embedding_cache_path, embeddings)

    # Dimensionality Reduction (384-D -> 50-D)
    # This pushes clusters apart and removes high-dimensional noise
    print("Applying UMAP for cluster separation...")
    reducer = UMAP(
        n_components=50, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    with open("data/umap_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    # Fuzzy Clustering on Reduced Space
    print(f"Fitting GMM on {config.N_CLUSTERS} clusters...")
    reduced_embeddings_64 = reduced_embeddings.astype("float64")

    gmm = GaussianMixture(
        n_components=config.N_CLUSTERS,
        covariance_type="diag",
        random_state=42,
        # Increase reg_covar slightly more to 1e-3 to prevent collapsed components
        reg_covar=1e-3,
        tol=1e-3,
        # Add n_init to help it find a stable starting point
        n_init=2,
    )

    # Fit using the high-precision data
    gmm.fit(reduced_embeddings_64)

    probs = gmm.predict_proba(reduced_embeddings)
    dominant_clusters = probs.argmax(axis=1)

    with open("data/gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    # Naming
    print("Generating distinct topic names...")
    cluster_names = get_cluster_topic_names(docs, dominant_clusters)

    # Persist to Vector DB
    # Store the 384-D embeddings for high-precision search,
    # but the distributions come from the UMAP-GMM pipeline.
    db = InternalVectorDB(embedding_dim=embeddings.shape[1])
    db.add_documents(embeddings, docs, distributions=probs.tolist())
    db.save(folder="data", cluster_names=cluster_names)

    print("\nPipeline complete. Reducer, GMM, and DB persisted.")


if __name__ == "__main__":
    main()
