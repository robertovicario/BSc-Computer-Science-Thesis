from sklearn.neural_network import MLPClassifier
from utils import X_test, X_train, y_test, y_train
from utils import evaluate_model, tune_hyperparameters

#
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

#
evaluate_model(mlp, X_train, y_train, X_test, y_test, cv=2)