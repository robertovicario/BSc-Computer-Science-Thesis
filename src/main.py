import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

def preprocess_data(file_path):
    #
    data = pd.read_csv(file_path)

    #
    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    data['condition'] = data['condition'].map(condition_encoding)

    #
    X = data.iloc[:, :-2]
    y = data.iloc[:, -1]

    #
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    #
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_standardized)

    return X, X_pca, y



data = pd.read_csv('./data/train.csv')
pd.set_option('display.max_columns', None)
print(data.describe(include='all'))