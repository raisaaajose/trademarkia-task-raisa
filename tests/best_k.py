import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from umap import UMAP
import os


def find_optimal_k(data, max_k=30):
    """
    Evaluates the best number of clusters by analyzing BIC and Silhouette scores
    within the UMAP manifold defined in clustering.py.
    """
    os.makedirs("plots", exist_ok=True)

    # Dimensionality Reduction updated for Fuzzy Logic requirements
    # We use 10-D and high min_dist to reflect the overlapping nature of the data
    print(f"Reducing {data.shape[0]} samples to 10-D 'Fuzzy' UMAP space...")
    reducer = UMAP(
        n_components=10, n_neighbors=100, min_dist=0.8, metric="cosine", random_state=42
    )
    reduced_data = reducer.fit_transform(data).astype("float64")

    n_components = range(2, max_k + 1, 2)
    bics = []
    silhouettes = []

    print("\nStarting K-Selection Sweep")
    print("-" * 45)

    for n in n_components:
        # Using n_init=2 to ensure we find a stable global optimum for BIC
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            random_state=42,
            reg_covar=1e-2,
            n_init=2,
        )

        labels = gmm.fit_predict(reduced_data)

        # Calculate Bayesian Information Criterion
        current_bic = gmm.bic(reduced_data)
        bics.append(current_bic)

        # Calculate Silhouette Score
        current_silhouette = silhouette_score(
            reduced_data,
            labels,
            sample_size=min(5000, len(reduced_data)),
            random_state=42,
        )
        silhouettes.append(current_silhouette)

        print(
            f"K={n:2d} | BIC={current_bic:12.2f} | Silhouette={current_silhouette:.4f}"
        )

    # Plotting
    print("\nGenerating visualization...")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("BIC Score ", color="tab:blue", fontsize=12)
    ax1.plot(n_components, bics, color="tab:blue", marker="o", linewidth=2, label="BIC")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:red", fontsize=12)
    ax2.plot(
        n_components,
        silhouettes,
        color="tab:red",
        marker="s",
        linewidth=2,
        label="Silhouette",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Evidence-Based K-Selection: BIC and Silhouette ", fontsize=14, pad=20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", frameon=True)

    output_path = "plots/k_selection_umap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Analysis plot saved to: {output_path}")


if __name__ == "__main__":
    data_path = "data/raw_embeddings.npy"
    if os.path.exists(data_path):
        raw_embeddings = np.load(data_path)
        find_optimal_k(raw_embeddings)
    else:
        print(f"Error: {data_path} not found. Run preprocessing first.")
