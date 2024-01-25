from sklearn.ensemble import RandomForestRegressor
from preprocessing import X_train, y_train
from tuning import tune_hyperparameters

random_forest = RandomForestRegressor()

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

best_estimator = tune_hyperparameters(random_forest, param_dist, X_train, y_train, cv=3)