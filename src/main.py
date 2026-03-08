import uvicorn
import pickle
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src.config import config

from src.cache import SemanticCache
from src.vector_store import InternalVectorDB

app = FastAPI(title="Trademarkia AI Semantic Search")

# Global State Management
# Initialized once to ensure high performance and low latency 
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DB = InternalVectorDB()
CACHE = SemanticCache(
    threshold=config.CACHE_THRESHOLD, capacity_per_cluster=config.CACHE_CAPACITY
)
GMM_MODEL = None
UMAP_REDUCER = None


@app.on_event("startup")
async def startup_event():
    """Load persistent models and vector database into memory on startup."""
    global GMM_MODEL, UMAP_REDUCER
    data_dir = "data"
    gmm_path = os.path.join(data_dir, "gmm_model.pkl")
    umap_path = os.path.join(data_dir, "umap_reducer.pkl")

    try:
        VECTOR_DB.load(data_dir)

        if os.path.exists(gmm_path):
            with open(gmm_path, "rb") as f:
                GMM_MODEL = pickle.load(f)
        if os.path.exists(umap_path):
            with open(umap_path, "rb") as f:
                UMAP_REDUCER = pickle.load(f)
            print("GMM and UMAP Reducer loaded successfully.")
    except Exception as e:
        print(f"Startup Error: {str(e)}")


# Request/Response Schemas
class QueryRequest(BaseModel):
    query: str


# API Endpoints
@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    1. Embeds query. 
    2. Projects query into UMAP manifold.
    3. Calculates top fuzzy clusters. 
    4. Performs multi-cluster semantic cache lookup. 
    5. On miss: Searches Vector DB and updates cache. 
    """
    query_text = request.query
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if GMM_MODEL is None or UMAP_REDUCER is None:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Run preprocessing."
        )

    # Generate Embedding (384-D)
    query_vec_384d = MODEL.encode(query_text).reshape(1, -1)
    query_vec_1d = query_vec_384d.flatten()

    # Transform to UMAP Space (Reduced 50-D) 
    # Ensures the query matches the space the GMM was trained on.
    query_vec_reduced = UMAP_REDUCER.transform(query_vec_384d)

    # Determine Fuzzy Cluster Probabilities 
    probs_all = GMM_MODEL.predict_proba(query_vec_reduced)[0]
    
    # Identify Top 3 clusters to support fuzzy search logic 
    top_indices = probs_all.argsort()[-3:][::-1]
    top_cluster_ids = [int(idx) for idx in top_indices]
    dom_cluster_id = top_cluster_ids[0]

    # Fuzzy Semantic Cache Lookup
    # Pass all three likely clusters to find similar queries 
    hit, cached_result = CACHE.query(query_text, query_vec_1d, top_cluster_ids)
    if hit:
        return cached_result

    # Cache Miss - Search the Internal Vector DB (using 384-D) 
    search_results = VECTOR_DB.search(query_vec_1d, k=1)

    if not search_results:
        return {
            "query": query_text,
            "cache_hit": False,
            "result": "No relevant documents found.",
            "dominant_cluster": dom_cluster_id,
        }

    final_result = search_results[0]["text"]
    similarity_score = float(search_results[0]["score"])

    # Prepare detailed cluster metadata for the response
    top_clusters_metadata = [
        {
            "id": int(idx),
            "score": round(float(probs_all[idx]), 4),
            "topic": VECTOR_DB.cluster_names.get(int(idx), f"Cluster {idx}"),
        }
        for idx in top_indices
    ]

    # Update Cache with result and its primary cluster
    CACHE.update(query_text, query_vec_1d, final_result, dom_cluster_id)

    return {
        "query": query_text,
        "cache_hit": False,
        "result": final_result,
        "similarity_score": similarity_score,
        "dominant_cluster": dom_cluster_id,
        "cluster_topic": top_clusters_metadata[0]["topic"],
        "fuzzy_logic": {
            "top_3_matches": top_clusters_metadata,
            "confidence_gap": round(
                top_clusters_metadata[0]["score"] - top_clusters_metadata[1]["score"], 4
            ),
        },
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """Returns the current state and performance of the semantic cache."""
    return CACHE.get_stats()


@app.delete("/cache")
async def clear_cache():
    """Flushes the cache from memory and deletes persistence file."""
    CACHE.flush()
    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    # Start the service on port 8000
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)