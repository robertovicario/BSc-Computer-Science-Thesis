from sklearn.cluster import KMeans
from unsupervised import X_test, evaluate_clustering

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_test)
labels = kmeans.predict(X_test)

evaluate_clustering(X_test, labels)