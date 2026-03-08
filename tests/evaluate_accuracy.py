import sys
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import aggressive_clean

def evaluate():
    # Load Original Data
    print("Loading original labels and data...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Align Labels with Cleaning Logic
    # Evaluate documents that survived the preprocessing filter
    print("Aligning ground-truth labels with cleaned dataset indices...")
    keep_indices = [i for i, doc in enumerate(newsgroups.data) if aggressive_clean(doc) is not None]
    true_labels = newsgroups.target[keep_indices]

    # Load Models and Generate Predictions
    try:
        print("Loading UMAP Reducer and GMM Model...")
        with open("data/umap_reducer.pkl", "rb") as f:
            reducer = pickle.load(f)
        with open("data/gmm_model.pkl", "rb") as f:
            gmm = pickle.load(f) 
            
        # Load the raw 384-D embeddings generated during preprocess
        if not os.path.exists("data/raw_embeddings.npy"):
            print("Error: raw_embeddings.npy not found. Run src/preprocess.py first")
            return
        
        embeddings_384d = np.load("data/raw_embeddings.npy")
        
        # Transform to the UMAP space
        print("Transforming embeddings to UMAP manifold...")
        embeddings_reduced = reducer.transform(embeddings_384d)
        
        # Predict clusters using the GMM trained on reduced space
        print("Predicting clusters...")
        # predict_proba for fuzzy or predict for hard assignment
        predicted_clusters = gmm.predict(embeddings_reduced)
        
    except FileNotFoundError as e:
        print(f"Error: Missing necessary model files in /data. {e}")
        return
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return

    # Consistency Check
    if len(true_labels) != len(predicted_clusters):
        print(f"Mismatch! True Labels: {len(true_labels)}, Predictions: {len(predicted_clusters)}")
        return

    # Calculate Metrics
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    nmi = normalized_mutual_info_score(true_labels, predicted_clusters)

    print(f"\n" + "="*30)
    print(f"--- Alignment Metrics ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print("="*30 + "\n")

    # Generate Normalized Heatmap
    # Normalizing by row ensures we see the % of each newsgroup in a cluster
    print("Generating category alignment heatmap...")
    plt.figure(figsize=(16, 12))
    cm = confusion_matrix(true_labels, predicted_clusters)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=False, cmap='YlGnBu', 
                xticklabels=range(len(np.unique(predicted_clusters))), 
                yticklabels=newsgroups.target_names)
    
    plt.title('Heatmap: Ground Truth vs. 14 Unsupervised Clusters')
    plt.xlabel('14 Clusters')
    plt.ylabel('Original 20 Newsgroups Categories')
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/category_alignment.png")
    print("Heatmap successfully saved to plots/category_alignment.png")

if __name__ == "__main__":
    evaluate()