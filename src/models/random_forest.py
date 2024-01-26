from sklearn.ensemble import RandomForestClassifier
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classifier

random_forest = RandomForestClassifier(
    criterion='gini',
    max_depth=8,
    max_features='log2',
    n_estimators=4
)
random_forest.fit(X_train, y_train)

evaluate_classifier(random_forest, X_train, y_train, X_test, y_test, cv=10)