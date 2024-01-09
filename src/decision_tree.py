from sklearn.tree import DecisionTreeClassifier
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
decision_tree = DecisionTreeClassifier(
    splitter='best',
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_depth=20,
    criterion='gini'
)
decision_tree.fit(X_train, y_train)

#
evaluate_model(decision_tree, X_train, y_train, X_test, y_test, cv=3)