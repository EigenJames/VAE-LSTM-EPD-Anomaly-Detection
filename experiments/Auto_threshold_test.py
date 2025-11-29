import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D

# Load the pickle file
trace_data = pd.read_pickle('D:/James_archive/OneDrive/On_Going/VASSCAA_submission/Paper draft/data copy/trace_data.pkl')

# Normalize the data (if there's no 'Unnamed: 0', directly normalize the entire dataset)
data_normalized = StandardScaler().fit_transform(trace_data)

# Convert back to a DataFrame for further use
normalized_trace_df = pd.DataFrame(data_normalized, columns=trace_data.columns)

# ----------- PCA Part with 3D Visualization and Cluster Labels ----------------
# Apply PCA to reduce the dimensionality to 3 components for 3D visualization
pca = PCA(n_components=3)
pca_data = pca.fit_transform(normalized_trace_df)

# Apply K-Means clustering with the optimal number of clusters
optimal_clusters = 3  # Based on the elbow method or prior knowledge
kmeans = KMeans(n_clusters=optimal_clusters)
clusters = kmeans.fit_predict(normalized_trace_df)

# Plot the 3D clustering result in the reduced PCA space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis', alpha=0.7)

# Add legend and colorbar to indicate which color belongs to which cluster
plt.title("K-Means Clustering in 3D PCA Space")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

# Add a color bar for the clusters
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Cluster Label')
plt.show()

# ----------- Time-Based Cluster Transitions and Final Etching Endpoint ----------------
# Add a 'Time' column (assuming the index represents the time sequence)
trace_data['Time'] = trace_data.index  # Ensure Time column is created from the index
trace_data['Cluster'] = clusters  # Add the cluster information

# Identify the cluster that appears at the end of the process (latest in time)
final_time_points = trace_data['Time'].tail(100)  # Example: Last 100 time points
final_clusters = trace_data.loc[final_time_points.index, 'Cluster']
final_etching_cluster = final_clusters.mode()[0]  # Get the most frequent cluster in the final time points

# Plot to see how the clusters change over time and highlight the final etching endpoint
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

# ----------- Detect Critical Peaks Automatically ----------------
# Calculate the standard deviation of each peak (column) over time
std_dev = trace_data.std()

# Set a threshold for identifying critical peaks (e.g., peaks with highest variation)
std_threshold = std_dev.mean() + 1.5 * std_dev.std()  # Example: peaks that deviate more than 1.5 std from the mean

# Identify critical peaks as those with standard deviation greater than the threshold
critical_peaks = std_dev[std_dev > std_threshold].index.tolist()

# Output the identified critical peaks
print(f"Automatically Identified Critical Peaks: {critical_peaks}")

# ----------- Automatically Determine Thresholds for Critical Peaks ----------------
# We will set the threshold as 90% of the maximum intensity for each critical peak
peak_thresholds = {}
for peak in critical_peaks:
    peak_max = trace_data[peak].max()
    peak_thresholds[peak] = 0.9 * peak_max  # Set threshold at 90% of the maximum value

# Output the thresholds for each critical peak
print(f"Automatically Determined Thresholds for Critical Peaks: {peak_thresholds}")

# ----------- Detect the Etching Endpoint Using the Critical Peaks and Thresholds ----------------
# Create a boolean condition where all peak intensities must exceed their thresholds
etching_condition = np.ones(len(trace_data), dtype=bool)
for peak, threshold in peak_thresholds.items():
    etching_condition &= trace_data[peak] > threshold  # All peaks must exceed their threshold

# Assume the etching endpoint happens at the last time period where all critical peaks exceed their thresholds
etching_times = trace_data[etching_condition]['Time']

# Automatically identify the final etching endpoint (last time point that meets the criteria)
if not etching_times.empty:
    final_etching_time = etching_times.iloc[-1]
    print(f"Final Etching Endpoint Detected at Time: {final_etching_time}")
else:
    print("No Etching Endpoint Detected Based on the Current Criteria")

# ----------- Plotting Critical Peaks and Etching Endpoint ----------------
# Plot intensities of all critical peaks over time and add threshold lines
plt.figure(figsize=(10, 6))
for peak, threshold in peak_thresholds.items():
    plt.plot(trace_data['Time'], trace_data[peak], label=f'Intensity at {peak}')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold for {peak}')

# Plot the final etching endpoint (if detected)
if not etching_times.empty:
    plt.axvline(x=final_etching_time, color='red', linestyle='--', label='Etching Endpoint')

plt.title("Critical Peak Intensities Over Time with Etching Endpoint Highlighted")
plt.xlabel("Time")
plt.ylabel("Intensity")
plt.legend()
plt.show()