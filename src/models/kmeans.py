from sklearn.cluster import KMeans
from evaluation import evaluate_clustering
from unsupervised import X_train

subset_size = int(0.01 * len(X_train))
X_train = X_train[:subset_size]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
labels = kmeans.labels_

evaluate_clustering(kmeans, X_train, labels, cv=10)