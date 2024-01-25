from sklearn.ensemble import RandomForestRegressor
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_regressor

random_forest = RandomForestRegressor(
    bootstrap=False,
    max_depth=40,
    max_features='sqrt',
    min_samples_split=10,
    n_estimators=200
)
random_forest.fit(X_train, y_train)

evaluate_regressor(random_forest, X_train, y_train, X_test, y_test, cv=3)