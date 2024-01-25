from sklearn.neural_network import MLPClassifier
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classifier

mlp = MLPClassifier(
    activation='tanh',
    alpha=0.001,
    hidden_layer_sizes=(50, 100, 50),
    learning_rate='adaptive'
)
mlp.fit(X_train, y_train)

evaluate_classifier(mlp, X_train, y_train, X_test, y_test, cv=3)