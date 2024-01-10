from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from utils import X_train

# Create and fit the Birch model
birch = Birch(threshold=0.5, n_clusters=None)
birch.fit(X_train)

# Predict the cluster labels
labels = birch.predict(X_train)

# Evaluate the model
silhouette_avg = silhouette_score(X_train, labels)
print("Silhouette Score: ", silhouette_avg)