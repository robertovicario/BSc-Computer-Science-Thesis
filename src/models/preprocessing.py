import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    df['condition'] = df['condition'].map(condition_encoding)

    features = df.iloc[:, :-2]
    target = df.iloc[:, -1]

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    pca = PCA(n_components=0.95, random_state=42)
    pca_features = pca.fit_transform(standardized_features)

    return pca_features, target

X_train, y_train = preprocess_data('./data/train.csv')
X_test, y_test = preprocess_data('./data/test.csv')