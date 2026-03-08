import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

def profile_clusters():
    # Load Data and Models
    data_dir = "data"
    try:
        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
        
        # Load the saved reducer to maintain consistency with the API
        with open(os.path.join(data_dir, "umap_reducer.pkl"), "rb") as f:
            reducer = pickle.load(f)
            
        embeddings_384d = np.load(os.path.join(data_dir, "raw_embeddings.npy"))
        
        # Get labels and names from metadata
        distributions = np.array(meta["distributions"])
        cluster_ids = distributions.argmax(axis=1)
        cluster_names = meta.get("cluster_names", {})
    except FileNotFoundError as e:
        print(f"Error: Missing files in /data. Run preprocess.py first ({e})")
        return

    # Cluster Population Report
    df = pd.DataFrame({'cluster': cluster_ids})
    counts = df['cluster'].value_counts().sort_index()
    
    print("\n--- Cluster Population Report ---")
    print(f"{'ID':<5} | {'Count':<8} | {'Topic Name'}")
    print("-" * 60)
    for cid, count in counts.items():
        name = cluster_names.get(str(cid), cluster_names.get(cid, "N/A"))
        print(f"{cid:<5} | {count:<8} | {name}")

    # Dimensionality Reduction 
    print("\nGenerating 2D manifold for visualization...")
    # use the existing reducer but transform to 2D for the plot
    from umap import UMAP
    viz_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_vecs = viz_reducer.fit_transform(embeddings_384d)

    # Advanced Plotting
    plt.figure(figsize=(14, 10))
    
    # Use a high-quality colormap
    scatter = plt.scatter(
        reduced_vecs[:, 0], 
        reduced_vecs[:, 1], 
        c=cluster_ids, 
        cmap='Spectral', 
        alpha=0.6, 
        s=2
    )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("Semantic Landscape: UMAP Manifold Projection", fontsize=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Add labels for the largest clusters directly on the plot
    for cid in counts.index:
        # Calculate the median position of points in this cluster
        mask = cluster_ids == cid
        median_pos = np.median(reduced_vecs[mask], axis=0)
        plt.text(median_pos[0], median_pos[1], str(cid), 
                 fontsize=12, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/cluster_landscape_umap.png", dpi=300)
    print("Visualization saved to plots/cluster_landscape_umap.png")

if __name__ == "__main__":
    profile_clusters()