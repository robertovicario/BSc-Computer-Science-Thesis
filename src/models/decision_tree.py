from sklearn.tree import DecisionTreeClassifier
from supervised import X_test, X_train, evaluate_classification, y_test, y_train

decision_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    max_features='log2'
)
decision_tree.fit(X_train, y_train)

evaluate_classification(decision_tree, X_train, y_train, X_test, y_test, cv=10)