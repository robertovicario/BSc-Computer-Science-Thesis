import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data with three clusters
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Fit a Gaussian Mixture Model to the data using the EM algorithm
gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
gmm.fit(X)

# Predict cluster assignments for each data point
labels = gmm.predict(X)

# Access the estimated parameters of the Gaussian components
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Visualize the data and cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(means[:, 0], means[:, 1], marker='o', c='red', s=100, label='Cluster Centers')
plt.legend()
plt.title('GMM Clustering with EM Algorithm')
plt.show()
