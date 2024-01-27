from sklearn.mixture import GaussianMixture
from unsupervised import X_test, evaluate_clustering

gaussian_mixture = GaussianMixture(n_components=3)
gaussian_mixture.fit(X_test)
labels = gaussian_mixture.predict(X_test)

evaluate_clustering(X_test, labels)