import sys
import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from umap import UMAP
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import get_cluster_topic_names, perform_fuzzy_clustering
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

    print(f"Performing Fuzzy Clustering on {config.N_CLUSTERS} clusters...")
    # This calls your clustering.py which has the reg_covar=1e-1 and n_components=10
    probs, dominant_clusters, reducer, gmm = perform_fuzzy_clustering(
        embeddings, n_clusters=config.N_CLUSTERS
    )

    with open("data/umap_reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    with open("data/gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    # Naming
    print("Generating distinct topic names...")
    cluster_names = get_cluster_topic_names(docs, dominant_clusters)

    with open("data/cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=4)
    # Persist to Vector DB
    # Store the 384-D embeddings for high-precision search,
    # but the distributions come from the UMAP-GMM pipeline.
    db = InternalVectorDB(embedding_dim=embeddings.shape[1])
    db.add_documents(embeddings, docs, distributions=probs.tolist())
    db.save(folder="data", cluster_names=cluster_names)

    print("\nPipeline complete. Reducer, GMM, and DB persisted.")


if __name__ == "__main__":
    main()
