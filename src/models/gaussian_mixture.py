from sklearn.mixture import GaussianMixture
from unsupervised import X_train, evaluate_clustering

subset_size = int(0.01 * len(X_train))
X_train = X_train[:subset_size]

gaussian_mixture = GaussianMixture(n_components=3)
gaussian_mixture.fit(X_train)
labels = gaussian_mixture.predict(X_train)

evaluate_clustering(X_train, labels)