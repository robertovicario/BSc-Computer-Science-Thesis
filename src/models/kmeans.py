from sklearn.cluster import KMeans
from evaluation import evaluate_clustering
from preprocessing import X_train

subset_size = int(0.01 * len(X_train))
X_train_subset = X_train[:subset_size]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train_subset)
labels = kmeans.labels_

evaluate_clustering(X_train_subset, labels)