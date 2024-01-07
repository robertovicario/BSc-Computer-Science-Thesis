from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from preprocessing import preprocess_data

#
X_train, y_train = preprocess_data('./data/train.csv', False)
X_test, y_test = preprocess_data('./data/test.csv', False)

#
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='constant'
)
model.fit(X_train, y_train)

#
predictions = model.predict(X_test)
scores = cross_val_score(model, X_train, y_train, cv=10)
report = classification_report(y_test, predictions)

#
print(scores)
print(report)