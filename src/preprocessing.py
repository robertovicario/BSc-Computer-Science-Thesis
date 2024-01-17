from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    df['condition'] = df['condition'].map(condition_encoding)

    features = df.iloc[:, :-2]
    target = df.iloc[:, -1]

    Q1 = features.quantile(0.25)
    Q3 = features.quantile(0.75)
    IQR = Q3 - Q1
    features = features[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    
    positive_features = np.where(standardized_features > 0, standardized_features, standardized_features.min() + 1)
    pt = PowerTransformer(method='box-cox')
    transformed_features = pt.fit_transform(positive_features)

    pca = PCA(n_components=0.95, random_state=42)
    pca_features = pca.fit_transform(transformed_features)

    return pca_features, target

X_train, y_train = preprocess_data('./data/train.csv')
X_test, y_test = preprocess_data('./data/test.csv')