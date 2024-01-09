from sklearn.linear_model import LogisticRegression
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
logistic_regression = LogisticRegression(
    solver='saga',
    penalty='l1',
    max_iter=100,
    class_weight=None,
    C=0.01
)

#
logistic_regression.fit(X_train, y_train)
evaluate_model(logistic_regression, X_train, y_train, X_test, y_test, cv=3)