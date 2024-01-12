from sklearn.cluster import Birch
from utils import X_train, tune_hyperparameters, y_train

#
birch = Birch()
birch.fit(X_train)
labels = birch.labels_

param_dist = {
    'threshold': [0.1, 0.5, 1.0, 1.5, 2.0],
    'branching_factor': [10, 20, 30, 40, 50]
}

#
best_estimator = tune_hyperparameters(birch, param_dist, X_train, y_train, cv=3)