from sklearn.linear_model import LogisticRegression
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classifier

logistic_regression = LogisticRegression(
    solver='lbfgs',
    penalty='l2'
)
logistic_regression.fit(X_train, y_train)

evaluate_classifier(logistic_regression, X_train, y_train, X_test, y_test, cv=10)