import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import json
import sys
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def profile_clusters():
    data_dir = "data"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load Cluster Names
        names_path = os.path.join(data_dir, "cluster_names.json")
        if os.path.exists(names_path):
            with open(names_path, "r") as f:
                cluster_names = json.load(f)
        else:
            # Fallback: Many VectorDBs save names inside their own metadata
            print("Warning: cluster_names.json not found. Using numeric IDs.")
            cluster_names = {}
            
        # Load Raw Embeddings (384-D)
        embeddings_path = os.path.join(data_dir, "raw_embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError("raw_embeddings.npy missing. Run preprocess.py.")
        embeddings_384d = np.load(embeddings_path)
        
        # Load GMM Model
        with open(os.path.join(data_dir, "gmm_model.pkl"), "rb") as f:
            gmm = pickle.load(f)
            
        # Load Production Reducer (10-D)
        with open(os.path.join(data_dir, "umap_reducer.pkl"), "rb") as f:
            production_reducer = pickle.load(f)

        # Generate Assignments in the Production Space
        print("Projecting to 10-D Production Space...")
        reduced_10d = production_reducer.transform(embeddings_384d)
        probs = gmm.predict_proba(reduced_10d)
        cluster_ids = probs.argmax(axis=1)

    except Exception as e:
        print(f" Error: {str(e)}")
        return

    # Population Report 
    counts = pd.Series(cluster_ids).value_counts().sort_index()
    print("\n--- Cluster Population Report ---")
    print(f"{'ID':<4} | {'Count':<6} | {'Topic Name'}")
    print("-" * 50)
    for cid, count in counts.items():
        # Handle string vs int keys in JSON
        name = cluster_names.get(str(cid), cluster_names.get(cid, f"Cluster {cid}"))
        print(f"{cid:<4} | {count:<6} | {name}")

    # 2D Visualization 
    
    print("\nGenerating 2D 'Fuzzy' Landscape visualization...")
    viz_reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.5, random_state=42)
    reduced_2d = viz_reducer.fit_transform(embeddings_384d)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], 
                         c=cluster_ids, cmap='tab20', alpha=0.6, s=8)
    
    # Label cluster centers for clarity
    for cid in counts.index:
        mask = cluster_ids == cid
        center = np.median(reduced_2d[mask], axis=0)
        plt.annotate(str(cid), center, fontsize=12, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title("Semantic Search Landscape", fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    save_path = os.path.join(plots_dir, "cluster_landscape_umap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success! Visualization saved to {save_path}")

if __name__ == "__main__":
    profile_clusters()