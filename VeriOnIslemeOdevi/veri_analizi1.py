import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Orijinal veri seti:")
print(X.head())
print(X.describe())

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_log = np.log1p(X)

X_squared = X ** 2


def plot_distributions(data, title):
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(data.columns):
        plt.subplot(2, 2, i + 1)
        sns.histplot(data[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


plot_distributions(X, "Original Data Distribution")
plot_distributions(X_normalized, "Normalized Data Distribution")
plot_distributions(X_log, "Log-transformed Data Distribution")
plot_distributions(X_squared, "Squared Data Distribution")


def classify_and_evaluate(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nPerformance for {dataset_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))


classify_and_evaluate(X, y, "Original Dataset")
classify_and_evaluate(X_normalized, y, "Normalized Dataset")
classify_and_evaluate(X_log, y, "Log-transformed Dataset")
classify_and_evaluate(X_squared, y, "Squared Dataset")