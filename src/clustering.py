from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction import text
from umap import UMAP
import pandas as pd


def perform_fuzzy_clustering(embeddings, n_clusters=14):
    # Dimensionality Reduction
    reducer = UMAP(
        n_components=50, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    refined_embeddings = reducer.fit_transform(embeddings)

    # Fit GMM on the REDUCED space
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="diag", random_state=42
    )
    gmm.fit(refined_embeddings)

    # Predict Probabilities using the reduced space
    probs = gmm.predict_proba(refined_embeddings)
    dominant_clusters = probs.argmax(axis=1)

    return probs, dominant_clusters, reducer


def get_cluster_topic_names(documents, labels, n_top_words=5):
    # Combine all documents in a cluster into one giant string
    df = pd.DataFrame({"text": documents, "cluster": labels})
    mega_docs = df.groupby("cluster")["text"].apply(lambda x: " ".join(x)).tolist()
    cluster_ids = sorted(df["cluster"].unique())

    custom_stops = list(text.ENGLISH_STOP_WORDS) + [
        "just",
        "know",
        "like",
        "use",
        "think",
        "make",
        "really",
        "thanks",
        "people",
        "thing",
        "point",
        "believe",
        "question",
        "using",
        "article",
        "thought",
        "world",
        "group",
    ]

    tfidf = TfidfVectorizer(
        stop_words=custom_stops,
        sublinear_tf=True,
        max_df=0.4,  # If a word is in >40% of clusters, it's not unique
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z]{5,}\b",
    )

    # Fit on the 14 Mega-Documents
    tfidf_matrix = tfidf.fit_transform(mega_docs)
    feature_names = tfidf.get_feature_names_out()

    cluster_names = {}
    for i, cluster_id in enumerate(cluster_ids):
        row = tfidf_matrix.getrow(i).toarray().flatten()
        top_indices = row.argsort()[-n_top_words:][::-1]

        # Get words that actually have a score
        names = [feature_names[idx].title() for idx in top_indices if row[idx] > 0]
        cluster_names[int(cluster_id)] = (
            ", ".join(names) if names else f"Cluster {cluster_id}"
        )

    return cluster_names
