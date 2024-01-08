from sklearn.ensemble import RandomForestClassifier
from utils import X_test, X_train, evaluate_model, y_test, y_train

#
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

#
evaluate_model(random_forest, X_train, y_train, X_test, y_test, cv=2)