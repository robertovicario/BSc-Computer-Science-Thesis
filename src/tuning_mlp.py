from sklearn.neural_network import MLPClassifier
from utils import X_test, X_train, evaluate_model, tune_hyperparameters, y_test, y_train

#
mlp = MLPClassifier()

#
param_dist = {
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'logistic', 'tanh']
}

#
best_estimator = tune_hyperparameters(mlp, param_dist, X_train, y_train, cv=3)
evaluate_model(best_estimator, X_train, y_train, X_test, y_test, cv=3)