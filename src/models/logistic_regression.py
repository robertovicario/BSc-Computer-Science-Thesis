from sklearn.linear_model import LogisticRegression
from supervised import X_test, X_train, evaluate_classification, y_test, y_train

logistic_regression = LogisticRegression(
    solver='lbfgs',
    penalty='l2'
)
logistic_regression.fit(X_train, y_train)

evaluate_classification(logistic_regression, X_train, y_train, X_test, y_test, cv=10)