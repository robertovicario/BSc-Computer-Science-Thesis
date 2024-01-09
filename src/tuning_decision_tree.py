from sklearn.tree import DecisionTreeClassifier
from utils import X_train, tune_hyperparameters, y_train

#
decision_tree = DecisionTreeClassifier()

#
param_dist = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

#
best_estimator = tune_hyperparameters(decision_tree, param_dist, X_train, y_train, cv=3)