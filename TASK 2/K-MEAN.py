import os
# Set the environment variable to suppress the warning from joblib's loky backend
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace "4" with the number of cores you want to use

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Mall Customer dataset
file_path = "Mall_customers.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Display the first few rows and check column names
print("Dataset Preview:")
print(df.head())
print("\nAvailable Columns:", df.columns.tolist())

# Check if necessary columns exist
expected_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}. Please check your dataset.")

# Selecting relevant features for clustering
data = df[expected_columns]

# Handle missing values by dropping rows with missing data
data = data.dropna()

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determine optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph to determine the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply K-Means with optimal K (assume K=5 based on the elbow graph, adjust as needed)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=df['Cluster'], palette='viridis')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Customer Segments')
plt.legend(title='Cluster')
plt.show()

# Save the clustered data to a CSV file
df.to_csv("clustered_customers.csv", index=False)
