from sklearn.neural_network import MLPClassifier
from utils import X_test, X_train, y_test, y_train
from utils import evaluate_model

#
mlp = MLPClassifier(
    solver='adam',
    learning_rate='adaptive',
    hidden_layer_sizes=(50, 100, 50),
    alpha=0.001,
    activation='tanh'
)

#
mlp.fit(X_train, y_train)
evaluate_model(mlp, X_train, y_train, X_test, y_test, cv=2)