from sklearn.tree import DecisionTreeRegressor
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_regression

decision_tree = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_features='log2'
)
decision_tree.fit(X_train, y_train)

evaluate_regression(decision_tree, X_train, y_train, X_test, y_test, cv=3)