import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import PowerTransformer

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    data['condition'] = data['condition'].map(condition_encoding)

    X = data.iloc[:, :-2]
    y = data.iloc[:, -1]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    X_positive = np.where(X_standardized > 0, X_standardized, X_standardized.min() + 1)
    pt = PowerTransformer(method='box-cox')
    X_transformed = pt.fit_transform(X_positive)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_transformed)

    return X_pca, y

X_train, y_train = preprocess_data('./data/train.csv')
X_test, y_test = preprocess_data('./data/test.csv')