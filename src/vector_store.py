import faiss
import numpy as np
import pickle
import os


class InternalVectorDB:
    def __init__(self, embedding_dim=384):
        # Choice of IndexFlatIP for Cosine Similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
        self.cluster_distributions = []  # Stores fuzzy assignments
        self.cluster_names = {}

    def add_documents(self, embeddings, texts, distributions=None):
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents.extend(texts)
        if distributions is not None:
            self.cluster_distributions.extend(distributions)

    def search(self, query_vector, k=14):
        distances, indices = self.index.search(
            np.array([query_vector]).astype("float32"), k
        )

        results = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                continue  # FAISS returns -1 if not enough results
            results.append(
                {
                    "text": self.documents[i],
                    "score": float(distances[0][j]),
                    "distribution": (
                        self.cluster_distributions[i]
                        if self.cluster_distributions
                        else None
                    ),
                }
            )
        return results

    def save(self, folder="data", cluster_names=None):
        os.makedirs(folder, exist_ok=True)
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(folder, "vector_db.faiss"))

        # Save the Metadata
        metadata = {
            "documents": self.documents,
            "distributions": self.cluster_distributions,
            "cluster_names": cluster_names,
        }
        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def load(self, folder="data"):
        index_path = os.path.join(folder, "vector_db.faiss")
        meta_path = os.path.join(folder, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.documents = meta["documents"]
                self.cluster_distributions = meta["distributions"]
                self.cluster_names = meta.get("cluster_names", {})
            return True
        return False
