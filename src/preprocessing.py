import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Separate features (X) and labels (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    """print(X.describe())"""  # Data before preprocessing

    # Encode 'condition' column
    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    data['condition'] = data['condition'].map(condition_encoding)

    # Standardize features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_standardized)

    """print(pd.DataFrame(X_pca).describe())""" # Data after preprocessing

    return X_pca, y

# Preprocess training and test data
X_train, y_train = preprocess_data('./data/training.csv')
X_test, y_test = preprocess_data('./data/test.csv')









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
