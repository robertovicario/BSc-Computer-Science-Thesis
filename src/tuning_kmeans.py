from sklearn.cluster import KMeans
from utils import X_train, tune_hyperparameters, y_train

#
kmeans = KMeans()
kmeans.fit(X_train)
labels = kmeans.labels_

param_dist = {
    'n_clusters': [2, 3, 4, 5],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [100, 200, 300],
    'random_state': [1000]
}

#
best_estimator = tune_hyperparameters(kmeans, param_dist, X_train, y_train, cv=3)