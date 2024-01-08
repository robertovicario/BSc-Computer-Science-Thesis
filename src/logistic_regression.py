from sklearn.linear_model import LogisticRegression
from utils import X_test, X_train, y_test, y_train
from utils import evaluate_model

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
logistic_regression.fit(X_train, y_train)
evaluate_model(logistic_regression, X_train, y_train, X_test, y_test, cv=2)