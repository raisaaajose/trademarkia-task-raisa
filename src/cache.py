import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.85,  # Tunable decision: Balance between precision and hit-rate
        capacity_per_cluster: int = 100,
        persistence_path: str = "data/cache_state.json",
    ):
        """
        Custom Semantic Cache built from first principles.
        Uses the GMM cluster structure to optimize lookup efficiency.
        """
        self.threshold = threshold
        self.capacity = capacity_per_cluster
        self.persistence_path = persistence_path

        # Structure: { cluster_id: OrderedDict({ query_text: {vector, result} }) }
        self.storage: Dict[int, OrderedDict] = {}

        # Analytics State
        self.hits = 0
        self.misses = 0
        self.total_entries = 0
        self.cluster_hits: Dict[int, int] = {}

        self.load_from_disk()

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Mathematical implementation of cosine similarity for vector comparison."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return (
            float(dot_product / (norm_v1 * norm_v2))
            if (norm_v1 > 0 and norm_v2 > 0)
            else 0.0
        )

    def query(
        self, 
        query_text: str, 
        query_vector: np.ndarray, 
        fuzzy_clusters: List[int]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Checks cache for semantic similarity across the top fuzzy cluster buckets.
        Reflects that documents can belong to multiple topics (e.g., Politics and Guns).
        """
        best_match_key = None
        highest_score = -1.0
        found_in_cluster = -1

        # Only search buckets relevant to the query's fuzzy distribution
        for cluster_id in fuzzy_clusters:
            cluster_entries = self.storage.get(cluster_id, OrderedDict())

            for q_text, data in cluster_entries.items():
                similarity = self._cosine_similarity(query_vector, np.array(data["vector"]))

                if similarity > self.threshold and similarity > highest_score:
                    highest_score = similarity
                    best_match_key = q_text
                    found_in_cluster = cluster_id

        if best_match_key:
            self.hits += 1
            self.cluster_hits[found_in_cluster] = self.cluster_hits.get(found_in_cluster, 0) + 1
            
            # LRU Logic: Move to end for the specific cluster bucket
            self.storage[found_in_cluster].move_to_end(best_match_key)

            match_data = self.storage[found_in_cluster][best_match_key]
            return True, {
                "query": query_text,
                "cache_hit": True,
                "matched_query": best_match_key,
                "similarity_score": round(highest_score, 4),
                "result": match_data["result"],
                "dominant_cluster": found_in_cluster,
            }

        self.misses += 1
        return False, None

    def update(
        self,
        query_text: str,
        query_vector: np.ndarray,
        result: str,
        dominant_cluster: int,
    ):
        """Adds new entry to the dominant cluster bucket with LRU eviction."""
        if dominant_cluster not in self.storage:
            self.storage[dominant_cluster] = OrderedDict()

        # Evict oldest entry if capacity for this cluster is reached
        if len(self.storage[dominant_cluster]) >= self.capacity:
            self.storage[dominant_cluster].popitem(last=False)
            self.total_entries -= 1

        self.storage[dominant_cluster][query_text] = {
            "vector": query_vector.tolist(),
            "result": result,
        }
        self.total_entries += 1
        self.save_to_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive cache state metrics for the /cache/stats endpoint."""
        total = self.hits + self.misses
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "cluster_performance": self.cluster_hits,
            "config": {
                "threshold": self.threshold,
                "capacity_per_cluster": self.capacity
            }
        }

    def flush(self):
        """Resets all state and deletes the persistence file on disk."""
        self.storage = {}
        self.hits = 0
        self.misses = 0
        self.total_entries = 0
        self.cluster_hits = {}
        if os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)

    def save_to_disk(self):
        """Persists cache state to JSON for state management across restarts."""
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(
                {
                    "storage": {k: list(v.items()) for k, v in self.storage.items()},
                    "hits": self.hits,
                    "misses": self.misses,
                    "total_entries": self.total_entries,
                    "cluster_hits": self.cluster_hits
                },
                f,
            )

    def load_from_disk(self):
        """Hydrates the cache from disk on startup."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r") as f:
                    data = json.load(f)
                    self.hits = data.get("hits", 0)
                    self.misses = data.get("misses", 0)
                    self.total_entries = data.get("total_entries", 0)
                    self.cluster_hits = data.get("cluster_hits", {})
                    
                    for k, v in data.get("storage", {}).items():
                        self.storage[int(k)] = OrderedDict(v)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                self.flush()