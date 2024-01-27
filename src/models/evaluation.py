import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, pairwise_distances, silhouette_score
from sklearn.model_selection import cross_val_score

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def evaluate_classification(model, X_train, y_train, X_test, y_test, cv):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print('cv_scores.mean:', f'{cv_scores.mean():.4f}')

    plot_confusion_matrix(y_test, y_pred)

def evaluate_clustering(X_train, labels):
    silhouette_avg = silhouette_score(X_train, labels)
    print('silhouette_score:', f'{silhouette_avg:.4f}')

    distance_matrix = pairwise_distances(X_train)
    sns.heatmap(distance_matrix, annot=False, cmap='viridis')
    plt.title('Pairwise Distance Matrix')
    plt.show()