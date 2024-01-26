from sklearn.tree import DecisionTreeClassifier
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classifier

decision_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    max_features='log2'
)
decision_tree.fit(X_train, y_train)

evaluate_classifier(decision_tree, X_train, y_train, X_test, y_test, cv=10)