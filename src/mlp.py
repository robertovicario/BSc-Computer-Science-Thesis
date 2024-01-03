from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Train MLP Model
def train_mlp_model(X_train, y_train):
    # Initialize the MLPClassifier
    # Here you can define the number of layers and the number of nodes in each layer among other parameters
    model = MLPClassifier(hidden_layer_sizes=(100,),  # Example: one hidden layer with 100 neurons
                          max_iter=300, 
                          activation='relu', 
                          solver='adam', 
                          random_state=42)
    
    # Train the model with the training data
    model.fit(X_train, y_train)
    
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    # Predict test data
    predictions = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, predictions)
    return report

# Assuming X_train, X_test, y_train, y_test are defined and preprocessed

# Train and evaluate the model
mlp_model = train_mlp_model(X_train, y_train)
print("MLP Model Training Complete.")

evaluation_report = evaluate_model(mlp_model, X_test, y_test)
print("MLP Model Evaluation Report:")
print(evaluation_report)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Implementing Cross-validation
def cross_validate_mlp_model(X, y, n_splits):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        model = train_mlp_model(X_train_fold, y_train_fold)  # Train model on fold
        predictions = model.predict(X_test_fold)  # Predict on validation fold
        accuracies.append(accuracy_score(y_test_fold, predictions))  # Calculate accuracy and append to list
    
    return np.mean(accuracies), np.std(accuracies)  # Return mean and standard deviation of accuracies

# Assuming X, y are your features and labels for the entire dataset
avg_accuracy, std_deviation = cross_validate_mlp_model(X_test, y_test, n_splits=10)
print(f"Cross-Validation Mean Accuracy: {avg_accuracy}, Standard Deviation: {std_deviation}")
