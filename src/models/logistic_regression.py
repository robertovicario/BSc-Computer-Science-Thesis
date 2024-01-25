from sklearn.linear_model import LogisticRegression
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classification

logistic_regression = LogisticRegression(
    C=0.01,
    max_iter=300,
    penalty='l1',
    solver='saga'
)
logistic_regression.fit(X_train, y_train)

evaluate_classification(logistic_regression, X_train, y_train, X_test, y_test, cv=3)