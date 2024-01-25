from sklearn.neural_network import MLPClassifier
from preprocessing import X_train, y_train
from tuning import tune_hyperparameters

mlp = MLPClassifier()

param_dist = {
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [(100,), (50, 50), (50, 100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'logistic', 'tanh']
}

best_estimator = tune_hyperparameters(mlp, param_dist, X_train, y_train, cv=3)