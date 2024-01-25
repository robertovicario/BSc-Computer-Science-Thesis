from sklearn.model_selection import RandomizedSearchCV

def tune_hyperparameters(model, param_dist, X, y, cv):
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)

    print('best_params_', random_search.best_params_)
    print('best_estimator_', random_search.best_estimator_)