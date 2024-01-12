import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    #
    data = pd.read_csv(file_path)

    #
    X = data.iloc[:, :-2]
    y = data.iloc[:, -1]

    #
    condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    data['condition'] = data['condition'].map(condition_encoding)

    #
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    #
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_standardized)

    return X, X_pca, y

def visualize_data(y):
    #
    label_counts = y.value_counts()

    #
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%')
    plt.title("Pie Chart")
    plt.show()

def evaluate_preprocessing(X):
    #
    df_pca = pd.DataFrame(data=X)
    corr_matrix = df_pca.corr()

    #
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title("Pearson's Correlation Matrix")
    plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test, cv):
    #
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    score = cross_val_score(model, X_train, y_train, cv=cv)

    #
    print(f"""Classification report:

    {report}
    """)
    print(f"""Cross-validation scores:

    {score}
    """)

    #
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def tune_hyperparameters(model, param_dist, X, y, cv):
    #
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)

    #
    best_params = random_search.best_params_
    best_estimator = random_search.best_estimator_

    print(f"""Best parameters:

    {best_params}
    """)
    
    return best_estimator

#
X0_train, X_train, y_train = preprocess_data('./data/train.csv')
X0_test, X_test, y_test = preprocess_data('./data/test.csv')