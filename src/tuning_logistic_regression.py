from sklearn.linear_model import LogisticRegression
from utils import X_train, tune_hyperparameters, y_train

#
logistic_regression = LogisticRegression()

#
param_dist = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500],
    'class_weight': [None, 'balanced']
}

#
best_estimator = tune_hyperparameters(logistic_regression, param_dist, X_train, y_train, cv=3)