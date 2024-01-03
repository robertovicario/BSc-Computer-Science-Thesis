import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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