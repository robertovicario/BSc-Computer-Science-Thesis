from matplotlib import pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import confusion_matrix, silhouette_score
from utils import X_train, y_train

# Create and fit the Birch model
birch = Birch(
    threshold=0.5,
    n_clusters=2
)
birch.fit(X_train)

# Predict the cluster labels
labels = birch.predict(X_train)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_train, labels)

# Plot the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y_train)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if i == j else 'black')

plt.show()

# Evaluate the model (Silhouette Score)
silhouette_avg = silhouette_score(X_train, labels)
print("Silhouette Score: ", silhouette_avg)