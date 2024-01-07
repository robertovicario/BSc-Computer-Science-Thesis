from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from preprocessing import preprocess_data

# Load and preprocess the data
X_train, y_train = preprocess_data('/kaggle/input/swell-heart-rate-variability-hrv/hrv dataset/data/final/train.csv', False)
X_test, y_test = preprocess_data('/kaggle/input/swell-heart-rate-variability-hrv/hrv dataset/data/final/test.csv', False)

# Initialize and train the Multi-layer Perceptron classifier
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='constant'
)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
report = classification_report(y_test, predictions)
score = cross_val_score(model, X_train, y_train, cv=3)

# Print evaluation metrics
print(f"""Classification report:
        
{report}
""")
print(f"""Cross-validation score:
        
{score}
""")