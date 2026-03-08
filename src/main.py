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
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DB = InternalVectorDB()
CACHE = SemanticCache(
    threshold=config.CACHE_THRESHOLD, capacity_per_cluster=config.CACHE_CAPACITY
)  # Built from first principles
GMM_MODEL = None
UMAP_REDUCER = None


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup for proper state management"""
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


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Accepts natural language queries, performs fuzzy clustering,
    and checks the semantic cache
    """
    query_text = request.query
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if GMM_MODEL is None or UMAP_REDUCER is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # Generate Embedding (384-D)
    query_vec_384d = MODEL.encode(query_text).reshape(1, -1)
    query_vec_1d = query_vec_384d.flatten()

    # Transform to UMAP Space and Determine Fuzzy Cluster Probabilities
    query_vec_reduced = UMAP_REDUCER.transform(query_vec_384d)
    probs_all = GMM_MODEL.predict_proba(query_vec_reduced)[0]

    # Temperature scaling for fuzzy distribution
    T = config.T
    smoothed_probs = np.exp(np.log(probs_all + 1e-10) / T)
    smoothed_probs /= np.sum(smoothed_probs)

    top_indices = smoothed_probs.argsort()[-3:][::-1]
    top_cluster_ids = [int(idx) for idx in top_indices]
    dom_cluster_id = top_cluster_ids[0]

    # Fuzzy Semantic Cache Lookup utilizing cluster structure
    hit, cached_result = CACHE.query(query_text, query_vec_1d, top_cluster_ids)
    if hit:
        # Returning required JSON structure for cache hit
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": cached_result["matched_query"],
            "similarity_score": cached_result["similarity_score"],
            "result": cached_result["result"],
            "dominant_cluster": dom_cluster_id,
        }

    # Cache Miss - Search the Internal Vector DB
    search_results = VECTOR_DB.search(query_vec_1d, k=1)
    if not search_results:
        return {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": 0.0,
            "result": "No relevant docs found.",
            "dominant_cluster": dom_cluster_id,
        }

    final_result = search_results[0]["text"]
    similarity_score = round(float(search_results[0]["score"]), 4)

    # Store the miss result in cache before returning
    CACHE.update(query_text, query_vec_1d, final_result, dom_cluster_id)

    # Returning required JSON structure for cache miss
    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": similarity_score,
        "result": final_result,
        "dominant_cluster": dom_cluster_id,
        "fuzzy_logic": {
            "top_3_distribution": [
                {"id": int(idx), "score": round(float(smoothed_probs[idx]), 4)}
                for idx in top_indices
            ]
        },
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """Returns current cache state statistics"""
    return CACHE.get_stats()


@app.delete("/cache")
async def clear_cache():
    """Flushes the cache entirely and resets all stats"""
    CACHE.flush()
    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
