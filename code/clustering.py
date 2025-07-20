import os
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def spectral_clustering(emb_path, year, norm = False, n_neigh_umap = 15, umap_components=5, umap_metric='cosine'):
    """
    works on my embeddings
    """
    df = pd.read_pickle(emb_path)
    df = df[df['year'] == year]
    print(f"Unique filtered cards printed in {year} are: {len(df)}")
    embeddings = np.array(df["embedding"].tolist())
    true_labels = df["colors"].str[0] # Extracts the first element of each list, which is the only element present

    if norm:
        embeddings = normalize(embeddings, norm='l2', axis=1)

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neigh_umap,
        n_components=umap_components,
        min_dist=0.0,
        metric=umap_metric,
        random_state=155      # To allow for reproducibility, no parallelism is applied.
        # n_jobs = -1         # Uses all available CPU cores.
    )
    embeddings = reducer.fit_transform(embeddings)

    print("Clustering the reduced data with Spectral Clustering...")
    spectral_cluster = SpectralClustering(
        n_clusters=5,
        affinity='rbf',
        random_state=155
    )

    cluster_labels = spectral_cluster.fit_predict(embeddings)

    perform_clustering_validation(true_labels, cluster_labels)

    print("Creating a 2D embedding for visualization...")
    vis_reducer = umap.UMAP(n_neighbors=n_neigh_umap, n_components=2, metric=umap_metric, random_state = 155)
    vis_embedding = vis_reducer.fit_transform(embeddings)

    # Create a DataFrame for plotting
    vis_df = pd.DataFrame(vis_embedding, columns=('x', 'y'))
    vis_df['clustered_labels'] = cluster_labels
    vis_df['true_labels'] = true_labels.values

    plot_results(vis_df)


def spectral_clustering_v2(emb_path, year, norm = False, n_neigh_umap = 12, umap_components=5, umap_metric='cosine'):
    """
    Works on embeddings in .parquet file
    """
    df = pd.read_parquet(emb_path)
    df = df[df['print_years'].apply(lambda years: year in years)]
    print(f"Unique filtered cards printed in {year} are: {len(df)}")
    embeddings = np.array(df["embedding"].tolist())
    true_labels = df["colors"].str[0] # Extracts the first element of each list, which is the only element present (monocol.parquet)

    if norm:
        embeddings = normalize(embeddings, norm='l2', axis=1)

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neigh_umap,
        n_components=umap_components,
        min_dist=0.0,
        metric=umap_metric,
        random_state=155      # To allow for reproducibility, no parallelism is applied.
        # n_jobs = -1         # Uses all available CPU cores.
    )
    embeddings = reducer.fit_transform(embeddings)

    print("Clustering the reduced data with Spectral Clustering...")
    spectral_cluster = SpectralClustering(
        n_clusters=5,
        affinity='rbf',
        random_state=155
    )

    cluster_labels = spectral_cluster.fit_predict(embeddings)

    perform_clustering_validation(true_labels, cluster_labels)

    print("Creating a 2D embedding for visualization...")
    vis_reducer = umap.UMAP(n_neighbors=n_neigh_umap, n_components=2, metric=umap_metric, random_state = 155)
    vis_embedding = vis_reducer.fit_transform(embeddings)

    # Create a DataFrame for plotting
    vis_df = pd.DataFrame(vis_embedding, columns=('x', 'y'))
    vis_df['clustered_labels'] = cluster_labels
    vis_df['true_labels'] = true_labels.values

    plot_results(vis_df)


def hdbscan_clustering(emb_path, year, norm = False, n_neigh_umap = 15, umap_components=5, umap_metric='cosine', hdbscan_min_cl_size=5, hdbscan_min_samples=11, hdbscan_eps=0.0):
    """
    Performs HDBSCAN clustering on card embeddings.
    Inputs:
      - emb_path (str) : path to the embeddings file
      - year (int) : this function will consider cards printed in the chosen year
      - norm (bool) : flag for normalization of vectors before clustering
      - n_neigh_umap (int) : the size of the neighborhood UMAP looks at when learning the manifold (lower-> local structure)
      - umap_components (int) : target dimension for dimensionality reduction preceeding clustering
      - umap_metric (str) : metric used by umap to compute distances
      - hdbscan_min_cl_size (int) : minimum number of samples in a group for that group to be considered a cluster
      - hdbscan_min_samples (int) : treshold for a point to be considered core point. defaults to min_cluster_size
      - hdbscan_eps (int) : A distance threshold. Clusters below this value will be merged
    """

    df = pd.read_pickle(emb_path)
    df = df[df['year'] == year]
    print(f"Unique filtered cards printed in {year} are: {len(df)}")
    embeddings = np.array(df["embedding"].tolist())
    true_labels = df["colors"].str[0] # Extracts the first element of each list, which is the only element present

    if norm:
        embeddings = normalize(embeddings, norm='l2', axis=1)

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neigh_umap,
        n_components=umap_components,
        min_dist=0.0,
        metric=umap_metric,
        random_state=155      # To allow for reproducibility, no parallelism is applied.
        # n_jobs = -1         # Uses all available CPU cores.
    )
    embeddings = reducer.fit_transform(embeddings)

    print("Clustering the reduced data with HDBSCAN...")
    clusterer = HDBSCAN(
        min_cluster_size = hdbscan_min_cl_size,
        min_samples = hdbscan_min_samples,
        cluster_selection_epsilon = hdbscan_eps,
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    perform_clustering_validation(true_labels, cluster_labels)

    print("Creating a 2D embedding for visualization...")
    vis_reducer = umap.UMAP(n_neighbors=n_neigh_umap, n_components=2, metric=umap_metric, random_state = 155)
    vis_embedding = vis_reducer.fit_transform(embeddings)

    # Create a DataFrame for plotting
    vis_df = pd.DataFrame(vis_embedding, columns=('x', 'y'))
    vis_df['clustered_labels'] = clusterer.labels_
    vis_df['true_labels'] = true_labels.values

    plot_results(vis_df)


def plot_results(vis_df):
    """
    Plots the results of clustering with matching colors,
    generalized so it can also manage multicolored and colorless cards.
    Expects vis_df to have the following columns:
      - 'x' : first dimension of data points
      - 'y' : second dimension of data points
      - 'true labels' : true label for each card
      - 'clustered_labels' : cluster label assigned by clustering algorithm
      
    """

    color_map = {
        'W': 'lightgrey',
        'U': 'blue',
        'B': 'black',
        'R': 'red',
        'G': 'green'
    }

    def get_color_category(label):
        if len(label) > 1:
            return 'Multicolor'
        elif label in ['C']:
            return 'Colorless'
        else:
            return label # W, U, B, R, G

    vis_df['color_category'] = vis_df['true_labels'].apply(get_color_category)

    plot_palette = {
        'W': 'lightgrey',
        'U': 'blue',
        'B': 'black',
        'R': 'red',
        'G': 'green',
        'Multicolor': 'gold',
        'Colorless': 'grey'
    }

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('UMAP Projection and Clustering of Card Embeddings', fontsize=16)

    # Plot 1: Colored by True Labels (Card Colors) 
    sns.scatterplot(
        data=vis_df,
        x='x',
        y='y',
        hue='color_category',
        palette=plot_palette,
        s=25,
        ax=ax1,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_title('2D UMAP Projection (Colored by True Card Color)')
    ax1.legend(title='Color Category')

    # Plot 2: Colored by Cluster Labels
    n_clusters = len(vis_df['clustered_labels'].unique()) - (1 if -1 in vis_df['clustered_labels'].unique() else 0)
    sns.scatterplot(data=vis_df, x='x', y='y', hue='clustered_labels', palette=sns.color_palette("deep", n_colors=n_clusters + 1), s=15, ax=ax2, alpha=0.7)
    ax2.set_title('2D UMAP Projection (Colored by Clustering Algorithm)')
    ax2.legend(title='Cluster ID')

    plt.show()


def perform_clustering_validation(true_labels, predicted_labels):
    """
    Computes and prints clustering validation metrics.
    Handles noise points, if present, by excluding them from the metrics

    Parameters:
    -----------
    true_labels : pandas.Series or np.ndarray
        The ground truth labels for the dataset.

    predicted_labels : np.ndarray
        The cluster labels generated by the clustering algorithm.
    """
    print("\n--- Clustering Validation Results ---")

    # Exclude points labelled by HDBSCAN as noise (-1), not usable for assignment comparisons
    noise_mask = (predicted_labels != -1)
    # If all points are noise, we can't calculate metrics
    if not np.any(noise_mask):
        print("All points were classified as noise. No validation metrics can be calculated.")
        print("--- End of Validation Report ---\n")
        return

    # Filter out the noise points for validation
    filtered_true_labels = true_labels[noise_mask]
    filtered_predicted_labels = predicted_labels[noise_mask]

    n_clusters = len(set(filtered_predicted_labels))
    n_noise = np.sum(predicted_labels == -1)
    
    print(f"Number of clusters found (excluding noise): {n_clusters}")
    print(f"Fraction of noise points: {n_noise} / {len(predicted_labels)}")

    # 1. Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(filtered_true_labels, filtered_predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

    # 2. Normalized Mutual Information (NMI)
    nmi_score = normalized_mutual_info_score(filtered_true_labels, filtered_predicted_labels)
    print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

    # 3. Homogeneity, Completeness, and V-Measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(filtered_true_labels, filtered_predicted_labels)
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"V-Measure: {v_measure:.4f}")

    # 4. Contingency Matrix
    con_matrix = contingency_matrix(filtered_true_labels, filtered_predicted_labels)
    print("\nContingency Matrix (rows=true, cols=predicted):")
    print(con_matrix)
    
    print("--- End of Validation Report ---\n")


if __name__ == "__main__":

    emb_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "all-MiniLM-L6-v2", "monocol_emb.pkl"))
    emb_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "all-mpnet-base-v2", "monocol_emb.pkl"))
    # df1 = pd.read_pickle(emb_path1)
    # df2 = pd.read_pickle(emb_path2)
    # X1 = np.array(df1["embedding"].tolist())
    # true_labels = df1["color_id"].str[0]

    spectral_clustering(emb_path2, 2016)

