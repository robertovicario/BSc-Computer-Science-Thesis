from sklearn.cluster import KMeans
from unsupervised import X_train, evaluate_clustering

subset_size = int(0.01 * len(X_train))
X_train = X_train[:subset_size]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
labels = kmeans.predict(X_train)

evaluate_clustering(X_train, labels)