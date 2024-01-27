from sklearn.cluster import Birch
from unsupervised import X_test, evaluate_clustering

birch = Birch(n_clusters=3)
birch.fit(X_test)
labels = birch.predict(X_test)

evaluate_clustering(X_test, labels)