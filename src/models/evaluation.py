import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.model_selection import cross_val_score

def evaluate_classification(model, X_train, y_train, X_test, y_test, cv):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print('cv_scores.mean:', f'{cv_scores.mean():.4f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def evaluate_clustering(model, X_train, labels, cv):
    silhouette_avg = silhouette_score(X_train, labels)
    print('silhouette_score:', f'{silhouette_avg:.4f}')

    cv_scores = cross_val_score(model, X_train, labels, cv=cv, scoring='silhouette')
    print('cv_scores.mean:', f'{cv_scores.mean():.4f}')

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