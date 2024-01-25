from sklearn.ensemble import RandomForestRegressor
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_regression

random_forest = RandomForestRegressor(
    n_estimators=50,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features='log2',
    max_depth=None,
    criterion='gini',
    bootstrap=False
)
random_forest.fit(X_train, y_train)

evaluate_regression(random_forest, X_train, y_train, X_test, y_test, cv=3)