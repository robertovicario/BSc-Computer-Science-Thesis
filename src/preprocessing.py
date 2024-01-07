import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(file_path, stats):
    # Load data from csv file into a DataFrame
    data = pd.read_csv(file_path)

    # Separate features (X) and target variable (y)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -1]

    # Map 'condition' column to numeric values for easier processing
    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    data['condition'] = data['condition'].map(condition_encoding)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Reduce dimensionality using PCA while retaining 95% of the variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_standardized)

    if (stats):
        # Print statistical summary of data before and after preprocessing
        print(f"""Train set before preprocessing:
              
        {pd.DataFrame(X).describe()}
        """)
        print(f"""Test set after preprocessing:
              
        {pd.DataFrame(X_pca).describe()}
        """)

    return X_pca, y

# Load and preprocess the data
preprocess_data('/kaggle/input/swell-heart-rate-variability-hrv/hrv dataset/data/final/train.csv', True)
preprocess_data('/kaggle/input/swell-heart-rate-variability-hrv/hrv dataset/data/final/test.csv', True)