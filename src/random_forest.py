from sklearn.ensemble import RandomForestClassifier
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
random_forest = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=40,
    criterion='entropy',
    bootstrap=True
)
random_forest.fit(X_train, y_train)

#
evaluate_model(random_forest, X_train, y_train, X_test, y_test, cv=3)