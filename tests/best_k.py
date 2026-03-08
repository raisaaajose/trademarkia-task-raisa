import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from umap import UMAP
import os


def find_optimal_k(data, max_k=30):
    """
    Evaluates the best number of clusters by analyzing BIC and Silhouette scores
    within a 50-dimensional UMAP manifold.
    """
    os.makedirs("plots", exist_ok=True)

    # Dimensionality Reduction (384D -> 50D)
    # Match the preprocessing logic exactly for the math to be valid.
    print(f"Reducing {data.shape[0]} samples to 50-D UMAP space...")
    reducer = UMAP(
        n_components=50, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    reduced_data = reducer.fit_transform(data).astype("float64")

    n_components = range(2, max_k + 1, 2)
    bics = []
    silhouettes = []

    print("\nStarting K-Selection Sweep...")
    print("-" * 40)

    for n in n_components:
        # Use reg_covar=1e-3 to prevent ill-defined covariance errors
        # caused by the high density of UMAP clusters.
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            random_state=42,
            reg_covar=1e-3,
            n_init=1,
        )

        # Fit and predict on the reduced manifold
        labels = gmm.fit_predict(reduced_data)

        # Calculate Bayesian Information Criterion
        current_bic = gmm.bic(reduced_data)
        bics.append(current_bic)

        # Calculate Silhouette Score
        # Higher score = better defined, more separated clusters.
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

    # Plotting the Dual-Axis Result
    print("\nGenerating visualization...")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # BIC Curve (Left Axis)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("BIC Score (Lower is Better)", color="tab:blue", fontsize=12)
    ax1.plot(n_components, bics, color="tab:blue", marker="o", linewidth=2, label="BIC")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # Silhouette Curve (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score (Higher is Better)", color="tab:red", fontsize=12)
    ax2.plot(
        n_components,
        silhouettes,
        color="tab:red",
        marker="s",
        linewidth=2,
        label="Silhouette",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(
        "GMM Selection: BIC vs. Silhouette (UMAP 50-D Space)", fontsize=14, pad=20
    )

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", frameon=True)

    output_path = "plots/k_selection_umap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Analysis plot saved to: {output_path}")


if __name__ == "__main__":
    # Ensure data exists before running
    data_path = "data/raw_embeddings.npy"
    if os.path.exists(data_path):
        raw_embeddings = np.load(data_path)
        find_optimal_k(raw_embeddings)
    else:
        print(
            f"Error: {data_path} not found. Please run preprocessing embedding step first."
        )
