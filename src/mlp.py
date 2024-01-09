from sklearn.neural_network import MLPClassifier
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
mlp = MLPClassifier(
    solver='adam',
    learning_rate='invscaling',
    hidden_layer_sizes=(100,),
    alpha=0.001,
    activation='logistic'
)
mlp.fit(X_train, y_train)

#
evaluate_model(mlp, X_train, y_train, X_test, y_test, cv=3)