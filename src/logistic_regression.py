from sklearn.linear_model import LogisticRegression
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
logistic_regression = LogisticRegression(

)

#
logistic_regression.fit(X_train, y_train)
evaluate_model(logistic_regression, X_train, y_train, X_test, y_test, cv=2)