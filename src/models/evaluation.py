import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def evaluate_classifier(model, X_train, y_train, X_test, y_test, cv=5):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('classification_report', report)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print('cv_scores.mean', f'{cv_scores.mean():.4f}')

    plot_confusion_matrix(y_test, y_pred)

def evaluate_regressor(model, X_train, y_train, X_test, y_test, cv=5):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print('r2_score', f'{r2:.4f}')

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    print('cv_scores.mean', f'{cv_scores.mean():.4f}')

    plot_confusion_matrix(y_test, y_pred)