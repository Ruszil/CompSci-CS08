import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

n_samples = 300
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42, cluster_std=0.60)

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
initial_centroids = kmeans.cluster_centers_
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
predicted_labels = knn.predict(X)


plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', marker='X', s=200, label='Initial Centroids')
plt.title('K-Nearest Neighbors Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()