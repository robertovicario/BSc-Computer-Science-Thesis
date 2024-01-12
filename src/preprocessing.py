from utils import X0_train, X_train, evaluate_preprocessing, visualize_data, y_train, y_test

#
visualize_data(y_train)
visualize_data(y_test)

#
evaluate_preprocessing(X0_train)
evaluate_preprocessing(X_train)