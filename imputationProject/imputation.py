# Gerekli kütüphaneleri import edelim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
import io

# Veriyi URL'den yükleyelim
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))

# Kullanacağımız özellikleri seçelim
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

# Veriyi hazırlayalım
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eksik verileri göster
print("Eksik veri analizi:")
print(X.isnull().sum())


# 1. Mean Imputation
def mean_imputation():
    imp_mean = SimpleImputer(strategy='mean')
    X_train_mean = pd.DataFrame(imp_mean.fit_transform(X_train), columns=X_train.columns)
    X_test_mean = pd.DataFrame(imp_mean.transform(X_test), columns=X_test.columns)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_mean, y_train)
    y_pred = rf.predict(X_test_mean)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


# 2. Hot Deck Imputation
# 2. Hot Deck Imputation
def hot_deck_imputation():
    from sklearn.neighbors import NearestNeighbors

    X_train_hotdeck = X_train.copy()
    X_test_hotdeck = X_test.copy()

    # Her sütun için eksik değerleri doldur
    for column in X_train_hotdeck.columns:
        if X_train_hotdeck[column].isnull().sum() > 0:  # Sadece eksik değeri olan sütunlar için çalış
            # Eksik olmayan verileri kullanarak NearestNeighbors modelini oluştur
            train_complete = X_train_hotdeck[~X_train_hotdeck[column].isnull()]
            train_missing = X_train_hotdeck[X_train_hotdeck[column].isnull()]

            # Diğer özellikleri kullan
            other_features = [f for f in X_train_hotdeck.columns if f != column]

            if len(train_complete) > 0 and len(train_missing) > 0:
                # NearestNeighbors modelini eğit
                nbrs = NearestNeighbors(n_neighbors=1).fit(train_complete[other_features])

                # En yakın komşuları bul
                distances, indices = nbrs.kneighbors(train_missing[other_features])

                # Eksik değerleri doldur
                X_train_hotdeck.loc[X_train_hotdeck[column].isnull(), column] = \
                    train_complete[column].iloc[indices.ravel()].values

    # Test seti için aynı işlemi yap
    for column in X_test_hotdeck.columns:
        if X_test_hotdeck[column].isnull().sum() > 0:
            test_missing = X_test_hotdeck[X_test_hotdeck[column].isnull()]
            other_features = [f for f in X_test_hotdeck.columns if f != column]

            if len(test_missing) > 0:
                # NearestNeighbors modelini eğit (train verisi üzerinde)
                train_complete = X_train_hotdeck[~X_train_hotdeck[column].isnull()]
                nbrs = NearestNeighbors(n_neighbors=1).fit(train_complete[other_features])

                # En yakın komşuları bul
                distances, indices = nbrs.kneighbors(test_missing[other_features])

                # Eksik değerleri doldur
                X_test_hotdeck.loc[X_test_hotdeck[column].isnull(), column] = \
                    train_complete[column].iloc[indices.ravel()].values

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_hotdeck, y_train)
    y_pred = rf.predict(X_test_hotdeck)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


# 3. MICE Imputation
def mice_imputation():
    imp_mice = IterativeImputer(random_state=42)
    X_train_mice = pd.DataFrame(imp_mice.fit_transform(X_train), columns=X_train.columns)
    X_test_mice = pd.DataFrame(imp_mice.transform(X_test), columns=X_test.columns)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_mice, y_train)
    y_pred = rf.predict(X_test_mice)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


# Tüm yöntemleri çalıştırıp sonuçları karşılaştıralım
print("\n1. Mean Imputation Sonuçları:")
mean_acc, mean_rep = mean_imputation()
print(f"Accuracy: {mean_acc}")
print(mean_rep)

print("\n2. Hot Deck Imputation Sonuçları:")
hotdeck_acc, hotdeck_rep = hot_deck_imputation()
print(f"Accuracy: {hotdeck_acc}")
print(hotdeck_rep)

print("\n3. MICE Imputation Sonuçları:")
mice_acc, mice_rep = mice_imputation()
print(f"Accuracy: {mice_acc}")
print(mice_rep)

# Sonuçları tablo halinde gösterelim
results_df = pd.DataFrame({
    'Yöntem': ['Mean Imputation', 'Hot Deck Imputation', 'MICE Imputation'],
    'Accuracy': [mean_acc, hotdeck_acc, mice_acc]
})

print("\nSonuçların Karşılaştırması:")
print(results_df)
# Sonuçları daha detaylı bir şekilde tablolama
detailed_results = pd.DataFrame({
   'Yöntem': ['Mean Imputation', 'Hot Deck Imputation', 'MICE Imputation'],
   'Accuracy': [mean_acc, hotdeck_acc, mice_acc],
   'Precision (Class 0)': [float(mean_rep.split()[4]), float(hotdeck_rep.split()[4]), float(mice_rep.split()[4])],
   'Recall (Class 0)': [float(mean_rep.split()[5]), float(hotdeck_rep.split()[5]), float(mice_rep.split()[5])],
   'F1-score (Class 0)': [float(mean_rep.split()[6]), float(hotdeck_rep.split()[6]), float(mice_rep.split()[6])],
   'Precision (Class 1)': [float(mean_rep.split()[9]), float(hotdeck_rep.split()[9]), float(mice_rep.split()[9])],
   'Recall (Class 1)': [float(mean_rep.split()[10]), float(hotdeck_rep.split()[10]), float(mice_rep.split()[10])],
   'F1-score (Class 1)': [float(mean_rep.split()[11]), float(hotdeck_rep.split()[11]), float(mice_rep.split()[11])]
})

# Tabloyu daha okunaklı hale getirmek için yuvarlama işlemi uygula
detailed_results_rounded = detailed_results.round(3)

print("\nDetaylı Performans Metrikleri Tablosu:")
print(detailed_results_rounded)

# DataFrame'i daha güzel göstermek için
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.width', None)        # Tablo genişliğini ayarla
print("\nDetaylı Performans Metrikleri Tablosu (Geliştirilmiş Görünüm):")
print(detailed_results_rounded.to_string(index=False))