import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D
import os

# Create a DataFrame to record the n_clusters and Final Etching Endpoint Detected at Time
results_df = pd.DataFrame(columns=['n_clusters', 'final_etching_time'])

def load_and_normalize_data(input_file):
    """ Load spectra data from a file and normalize it. """
    trace_data = pd.read_pickle(input_file)
    data_normalized = StandardScaler().fit_transform(trace_data)
    normalized_trace_df = pd.DataFrame(data_normalized, columns=trace_data.columns)
    trace_data['Time'] = trace_data.index  # Assume the index represents the time sequence
    return trace_data, normalized_trace_df

def plot_3d_pca_clusters(pca_data, clusters, n_clusters):
    """ Plot the 3D PCA clustering results with labeled clusters. """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis', alpha=0.7)

    # Add cluster labels
    for cluster in np.unique(clusters):
        cluster_points = pca_data[clusters == cluster]
        center = np.mean(cluster_points, axis=0)
        ax.text(center[0], center[1], center[2], f'Cluster {cluster}', fontsize=14, color='black', 
                fontweight='bold', backgroundcolor='white')

    plt.title(f"K-Means Clustering in 3D PCA Space ({n_clusters} Clusters)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.show()

def plot_cluster_transitions(trace_data, final_etching_cluster):
    """ Plot the cluster transitions over time with the final etching endpoint highlighted. """
    plt.figure(figsize=(10, 6))
    plt.scatter(trace_data['Time'], trace_data['Cluster'], c=trace_data['Cluster'], cmap='viridis', alpha=0.5)
    plt.title("Cluster Transitions Over Time with Final Etching Endpoint Highlighted")
    plt.xlabel("Time")
    plt.ylabel("Cluster")

    # Highlight the final etching endpoint cluster
    endpoint_times = trace_data[trace_data['Cluster'] == final_etching_cluster]['Time']
    plt.scatter(endpoint_times, np.full(len(endpoint_times), final_etching_cluster), color='red', label='Final Etching Endpoint', s=50)

    plt.legend()
    plt.show()

def plot_elbow_method(wcss):
    """ Plot the elbow method to determine the optimal number of clusters. """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(wcss)+1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def get_time_ranges(trace_data, clusters):
    """ Get the time range for each cluster and return a dictionary. """
    time_ranges = {}
    for cluster in np.unique(clusters):
        cluster_times = trace_data[trace_data['Cluster'] == cluster]['Time']
        time_range = (cluster_times.min(), cluster_times.max())
        time_ranges[cluster] = time_range
        print(f"Cluster {cluster}: Time from {time_range[0]} to {time_range[1]}")
    return time_ranges

def detect_critical_peaks(trace_data, std_multiplier=1.5):
    """ Detect critical peaks based on standard deviation threshold. """
    std_dev = trace_data.std()
    std_threshold = std_dev.mean() + std_multiplier * std_dev.std()
    critical_peaks = std_dev[std_dev > std_threshold].index.tolist()

    print(f"Automatically Identified Critical Peaks: {critical_peaks}")
    return critical_peaks

def determine_thresholds(earliest_spectrum, critical_peaks):
    """ Determine thresholds based on the earliest spectrum in the latest cluster. """
    peak_thresholds = {peak: earliest_spectrum[peak].values[0] for peak in critical_peaks}
    print(f"Stricter Thresholds for Critical Peaks: {peak_thresholds}")
    return peak_thresholds

def plot_critical_peaks_with_thresholds(trace_data, critical_peaks, peak_thresholds, final_etching_time, latest_cluster):
    """ Plot the intensities of all critical peaks over time and add threshold lines. """
    plt.figure(figsize=(10, 6))
    for peak, threshold in peak_thresholds.items():
        plt.plot(trace_data['Time'], trace_data[peak], label=f'Intensity at {peak}')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold for {peak} (Earliest Time in Cluster {latest_cluster})')

    # Plot the final etching endpoint (if detected)
    if final_etching_time is not None:
        plt.axvline(x=final_etching_time, color='red', linestyle='--', label='Etching Endpoint')

    plt.title(f"Critical Peak Intensities Over Time with Thresholds Based on Earliest Spectra in Cluster {latest_cluster} and Etching Endpoint")
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.show()

def PCA_cluster_EPD(input_file, n_clusters=3, offset=0):
    """
    Main function to analyze spectra data using PCA, K-Means clustering, and detect 
    the final etching endpoint based on the latest cluster.
    
    Parameters:
    - input_file: Path to the pickle file containing spectra data.
    - n_clusters: Number of clusters to use for K-Means clustering.
    """

    # Step 1: Load and normalize data
    trace_data, normalized_trace_df = load_and_normalize_data(input_file)

    # Step 2: Elbow Method for Optimal Clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(normalized_trace_df)
        wcss.append(kmeans.inertia_)
    
    plot_elbow_method(wcss)

    # Step 3: PCA and K-Means clustering
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(normalized_trace_df)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(normalized_trace_df)
    trace_data['Cluster'] = clusters

    # Step 4: Plot the 3D PCA clustering result
    plot_3d_pca_clusters(pca_data, clusters, n_clusters)

    # Step 5: Get time ranges for each cluster
    time_ranges = get_time_ranges(trace_data, clusters)

    # Step 6: Identify the latest cluster based on time
    latest_cluster = trace_data.groupby('Cluster')['Time'].max().idxmax()
    print(f"Automatically selected latest cluster for endpoint detection: Cluster {latest_cluster}")

    # Step 7: Plot cluster transitions over time
    plot_cluster_transitions(trace_data, latest_cluster)

    # Step 8: Get earliest spectrum in the latest cluster
    latest_cluster_data = trace_data[trace_data['Cluster'] == latest_cluster]
    earliest_time = latest_cluster_data['Time'].min() + offset
    earliest_spectrum = latest_cluster_data[latest_cluster_data['Time'] == earliest_time]
    
    # Silhouette Score (ranges from -1 to 1, higher is better)
    silhouette_avg = silhouette_score(normalized_trace_df, clusters)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    # Calinski-Harabasz Index (higher is better)
    calinski_harabasz = calinski_harabasz_score(normalized_trace_df, clusters)
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")

    # Davies-Bouldin Index (lower is better)
    davies_bouldin = davies_bouldin_score(normalized_trace_df, clusters)
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")


    # Step 9: Detect critical peaks and determine thresholds
    critical_peaks = detect_critical_peaks(trace_data)
    peak_thresholds = determine_thresholds(earliest_spectrum, critical_peaks)

    # Step 10: Detect the final etching endpoint
    etching_condition = np.ones(len(trace_data), dtype=bool)
    for peak, threshold in peak_thresholds.items():
        etching_condition &= trace_data[peak] > threshold  # All peaks must exceed the threshold based on earliest time point

    etching_times = trace_data[etching_condition]['Time']
    final_etching_time = etching_times.iloc[-1] if not etching_times.empty else None
    print(f"Final Etching Endpoint Detected at Time: {final_etching_time}")

    # Record the results in the results DataFrame
    global results_df
    results_df = pd.concat([results_df, pd.DataFrame({'n_clusters': [n_clusters], 'final_etching_time': [final_etching_time]})], ignore_index=True)

    # Step 11: Plot critical peak intensities with thresholds and etching endpoint
    plot_critical_peaks_with_thresholds(trace_data, critical_peaks, peak_thresholds, final_etching_time, latest_cluster)

# Example usage:
# analyze_spectra_data('path/to/your/spectra_data.pkl', n_clusters=5)

# Plot the results DataFrame at the end of the loop
def plot_results():
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_clusters'], results_df['final_etching_time'], marker='o')
    plt.title('Effect of n_clusters on Final Etching Endpoint Detected Time')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Final Etching Endpoint Time')
    plt.show()