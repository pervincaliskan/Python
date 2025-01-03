import matplotlib
matplotlib.use('TkAgg')  # Backend ayarı

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Veri setini yükleme
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# Veri seti hakkında temel bilgiler
print("Veri Seti Boyutu:", data.shape)
print("\nSınıf Dağılımı:")
print(data['quality'].value_counts().sort_index())

# Bağımlı ve bağımsız değişkenleri ayırma
X = data.drop('quality', axis=1)
y = data['quality']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Eğitim süreleri için dictionary
training_times = {}

# 1. Orijinal Veri ile Sınıflandırma
start_time = time.time()
rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_original.fit(X_train_scaled, y_train)
training_times['Original'] = time.time() - start_time

y_pred_original = rf_original.predict(X_test_scaled)
original_accuracy = accuracy_score(y_test, y_pred_original)

print("\n1. Orijinal Veri ile Sınıflandırma Sonuçları:")
print(classification_report(y_test, y_pred_original, zero_division=0))

# 2. PCA Uygulama
start_time = time.time()
pca = PCA(n_components=0.95)  # %95 varyans korunacak
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
training_times['PCA'] = time.time() - start_time

print(f"\nPCA sonrası boyut: {X_train_pca.shape[1]}")
print("Açıklanan varyans oranları:", pca.explained_variance_ratio_)

# PCA sonrası sınıflandırma
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
pca_accuracy = accuracy_score(y_test, y_pred_pca)

print("\n2. PCA sonrası Sınıflandırma Sonuçları:")
print(classification_report(y_test, y_pred_pca, zero_division=0))

# 3. LDA Uygulama
start_time = time.time()
n_components = len(np.unique(y)) - 1
lda = LDA(n_components=n_components)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)
training_times['LDA'] = time.time() - start_time

print(f"\nLDA sonrası boyut: {X_train_lda.shape[1]}")

# LDA sonrası sınıflandırma
rf_lda = RandomForestClassifier(n_estimators=100, random_state=42)
rf_lda.fit(X_train_lda, y_train)
y_pred_lda = rf_lda.predict(X_test_lda)
lda_accuracy = accuracy_score(y_test, y_pred_lda)

print("\n3. LDA sonrası Sınıflandırma Sonuçları:")
print(classification_report(y_test, y_pred_lda, zero_division=0))

# Görselleştirmeler

# 1. Doğruluk Oranları Karşılaştırması
plt.figure(figsize=(12, 6))
accuracies = {
    'Original': original_accuracy,
    'PCA': pca_accuracy,
    'LDA': lda_accuracy
}

plt.subplot(1, 2, 1)
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Doğruluk Oranları Karşılaştırması')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

# 2. Eğitim Süreleri Karşılaştırması
plt.subplot(1, 2, 2)
plt.bar(training_times.keys(), training_times.values())
plt.title('Eğitim Süreleri Karşılaştırması')
plt.ylabel('Süre (saniye)')

for i, v in enumerate(training_times.values()):
    plt.text(i, v + 0.01, f'{v:.3f}s', ha='center')

plt.tight_layout()
plt.show()

# 3. Özellik Önem Düzeyleri (Original)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_original.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Orijinal Özellik Önem Düzeyleri')
plt.show()

# 4. Confusion Matrix Karşılaştırması
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original
sns.heatmap(confusion_matrix(y_test, y_pred_original),
            annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Original Data Confusion Matrix')

# PCA
sns.heatmap(confusion_matrix(y_test, y_pred_pca),
            annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_title('PCA Confusion Matrix')

# LDA
sns.heatmap(confusion_matrix(y_test, y_pred_lda),
            annot=True, fmt='d', ax=axes[2], cmap='Blues')
axes[2].set_title('LDA Confusion Matrix')

plt.tight_layout()
plt.show()
