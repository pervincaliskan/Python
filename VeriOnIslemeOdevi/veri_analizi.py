# 1. Gerekli kütüphaneleri içe aktaralım
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Iris veri setini yüklemek için
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için
from sklearn.preprocessing import MinMaxScaler  # Normalizasyon için
from sklearn.svm import SVC  # Sınıflandırıcı model
from sklearn.metrics import accuracy_score, classification_report  # Performans ölçümü için

# 2. Iris veri setini yükleyelim
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Özellikler (sepal length, sepal width, vb.)
y = iris.target  # Hedef değişken (çiçek türleri: 0, 1, 2)

# 3. Orijinal veriyi inceleyelim
print("Orijinal veri seti:")
print(X.head())  # İlk 5 satırı göster
print(X.describe())  # İstatistiksel özet

# 4. Üç farklı dönüşüm uygulayalım:

# A. Normalizasyon (0-1 aralığına ölçeklendirme)
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# B. Logaritmik Dönüşüm
X_log = np.log1p(X)  # log(1+x) kullanıyoruz çünkü bazı değerler 0 olabilir

# C. Karesel Dönüşüm
X_squared = X ** 2


# 5. Sınıflandırma ve değerlendirme fonksiyonu
def classify_and_evaluate(X, y, dataset_name):
    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modeli oluştur ve eğit
    model = SVC()
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Sonuçları yazdır
    print(f"\nPerformance for {dataset_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))


# 6. Her veri seti için sınıflandırma yap
classify_and_evaluate(X, y, "Original Dataset")  # Orijinal veri
classify_and_evaluate(X_normalized, y, "Normalized Dataset")  # Normalize edilmiş veri
classify_and_evaluate(X_log, y, "Log-transformed Dataset")  # Logaritmik dönüşüm
classify_and_evaluate(X_squared, y, "Squared Dataset")  # Karesel dönüşüm