from sklearn.neural_network import MLPClassifier
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classifier

mlp = MLPClassifier()
mlp.fit(X_train, y_train)

evaluate_classifier(mlp, X_train, y_train, X_test, y_test, cv=3)