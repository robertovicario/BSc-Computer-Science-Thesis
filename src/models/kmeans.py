from sklearn.cluster import KMeans
from evaluation import evaluate_clustering
from preprocessing import X_train

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
labels = kmeans.labels_

evaluate_clustering(X_train, labels)