"""

LogisticRegression

"""

# Train Decision Tree Model
def train_logistic_regression_model(X_train, y_train):
    # Initialize the Logisting Regression Classifier
    model = LogisticRegression(random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model

"""

DecisionTreeClassifier

"""

# Train Decision Tree Model
def train_decision_tree_model(X_train, y_train):
    # Initialize the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model

"""

RandomForestClassifier

"""

# Train Random Forest Model
def train_random_forest_model(X_train, y_train):
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model