from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import X_train

#
kmeans = KMeans(
    random_state=1000,
    n_init=20,
    n_clusters=3,
    max_iter=100,
    init='k-means++'
)
kmeans.fit(X_train)
labels = kmeans.labels_

#
silhouette_avg = silhouette_score(X_train, labels)
print("silhouette_score:", silhouette_avg)