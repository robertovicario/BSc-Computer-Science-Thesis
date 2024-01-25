import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(file_path):
    df = pd.read_csv(file_path)
    pd.set_option('display.max_columns', None)
    print(df.describe(include='all'))

    numeric_data = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.show()

    condition_counts = df['condition'].value_counts()
    labels = condition_counts.index
    sizes = condition_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.show()

analyze_data('./data/train.csv')