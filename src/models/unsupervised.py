import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    df['condition'] = df['condition'].map(condition_encoding)

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(df)

    pca = PCA(n_components=0.95, random_state=42)
    pca_features = pca.fit_transform(standardized_features)

    return pca_features

def evaluate_clustering(X_train, labels):
    silhouette_avg = silhouette_score(X_train, labels)
    print('silhouette_score:', f'{silhouette_avg:.4f}')

    silhouette_vals = silhouette_samples(X_train, labels)
    y_ticks = []
    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(set(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)
        color = plt.cm.viridis(float(i) / len(set(labels)))

        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks(y_ticks, range(len(set(labels))))
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.title('Silhouette Plot')
    plt.show()

X_train = preprocess_data('./data/train.csv')