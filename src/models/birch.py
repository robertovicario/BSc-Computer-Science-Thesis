from sklearn.cluster import Birch
from unsupervised import X_train, evaluate_clustering

subset_size = int(0.01 * len(X_train))
X_train = X_train[:subset_size]

birch = Birch(n_clusters=3)
birch.fit(X_train)
labels = birch.predict(X_train)

evaluate_clustering(X_train, labels)