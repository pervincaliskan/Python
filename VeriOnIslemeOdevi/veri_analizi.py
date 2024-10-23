# 1. Kütüphaneleri içe aktarma
import pandas as pd  # Veri işleme için
import numpy as np  # Matematiksel işlemler için
from sklearn.datasets import load_iris  # Hazır iris veri setini yüklemek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için
from sklearn.preprocessing import MinMaxScaler  # Normalizasyon için
from sklearn.svm import SVC  # Sınıflandırma modeli
from sklearn.metrics import accuracy_score, classification_report  # Başarı ölçümü için
import matplotlib.pyplot as plt  # Grafik çizmek için
import seaborn as sns  # Grafikleri güzelleştirmek için

# 2. Veri setini yükleme
iris = load_iris()  # Iris veri setini yükle
# X: özellikler (sepal length, sepal width, petal length, petal width)
# y: çiçek türleri (0: setosa, 1: versicolor, 2: virginica)
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 3. Orijinal veriyi inceleyelim
print("Orijinal veri seti:")
print(X.head())  # İlk 5 satırı göster
print(X.describe())  # İstatistiksel özet

# 4. Veri dönüşümlerini uygulama
# A. Normalizasyon: Tüm değerleri 0-1 aralığına getirme
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# B. Logaritmik dönüşüm: Büyük değerleri küçültme
X_log = np.log1p(X)

# C. Karesel dönüşüm: Değerlerin karesi
X_squared = X ** 2


# 5. Her dönüşüm için grafik çizme
def plot_distributions(data, title):
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(data.columns):
        plt.subplot(2, 2, i + 1)  # 2x2'lik grid oluştur
        sns.histplot(data[column], kde=True)  # Histogram çiz
        plt.title(column)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


# Her veri seti için dağılım grafiklerini çiz
plot_distributions(X, "Original Data Distribution")
plot_distributions(X_normalized, "Normalized Data Distribution")
plot_distributions(X_log, "Log-transformed Data Distribution")
plot_distributions(X_squared, "Squared Data Distribution")


# 6. Sınıflandırma ve değerlendirme fonksiyonu
def classify_and_evaluate(X, y, dataset_name):
    # Veriyi %70 eğitim, %30 test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Sınıflandırma modelini oluştur ve eğit
    model = SVC()
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)

    # Sonuçları yazdır
    print(f"\nPerformance for {dataset_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))


# 7. Her veri seti için sınıflandırma yap
classify_and_evaluate(X, y, "Original Dataset")
classify_and_evaluate(X_normalized, y, "Normalized Dataset")
classify_and_evaluate(X_log, y, "Log-transformed Dataset")
classify_and_evaluate(X_squared, y, "Squared Dataset")