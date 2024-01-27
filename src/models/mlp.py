from sklearn.neural_network import MLPClassifier
from preprocessing import X_test, X_train, y_test, y_train
from evaluation import evaluate_classification

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Adjust the hidden layer sizes as needed
    max_iter=500,                  # Maximum number of iterations
    random_state=42,               # Random seed for reproducibility
    alpha=0.0001,                  # L2 regularization parameter
    activation='relu',             # Activation function for hidden layers
    solver='adam',                 # Optimizer
    batch_size=200,                # Batch size
    early_stopping=True,           # Enable early stopping
    validation_fraction=0.1,       # Fraction of training data used for validation
    n_iter_no_change=10,           # Number of epochs with no improvement to trigger early stopping
    tol=1e-4,                      # Tolerance to declare convergence
    verbose=True
)
mlp.fit(X_train, y_train)

evaluate_classification(mlp, X_train, y_train, X_test, y_test, cv=3)