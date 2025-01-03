import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LASSO için özellik seçimi
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
print("LASSO Özellik Katsayıları:", lasso.coef_)

# Önemli özellikleri seç (LASSO)
lasso_selected_features = np.where(abs(lasso.coef_) > 0)[0]
print("\nLASSO ile seçilen özellikler:", lasso_selected_features)

# Elastic Net için özellik seçimi
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
print("\nElastic Net Özellik Katsayıları:", elastic.coef_)

# Önemli özellikleri seç (Elastic Net)
elastic_selected_features = np.where(abs(elastic.coef_) > 0)[0]
print("\nElastic Net ile seçilen özellikler:", elastic_selected_features)


# Seçilen özelliklerle sınıflandırma performansını değerlendir
def evaluate_features(X_train, X_test, y_train, y_test, selected_features, method_name):
    # Seçilen özelliklerle yeni veri setleri oluştur
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Lojistik Regresyon modeli
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_selected, y_train)

    # Tahminler
    y_pred = clf.predict(X_test_selected)

    # Sonuçları yazdır
    print(f"\n{method_name} ile seçilen özelliklerle sınıflandırma sonuçları:")
    print("Doğruluk:", accuracy_score(y_test, y_pred))
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))


# LASSO ve Elastic Net sonuçlarını değerlendir
evaluate_features(X_train_scaled, X_test_scaled, y_train, y_test, lasso_selected_features, "LASSO")
evaluate_features(X_train_scaled, X_test_scaled, y_train, y_test, elastic_selected_features, "Elastic Net")