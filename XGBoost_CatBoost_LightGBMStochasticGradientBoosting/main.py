import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

# Veri setini yükle
iris = load_iris()
X, y = iris.data, iris.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri oluştur (default parametrelerle)
models = {
    'XGBoost': XGBClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
    'LightGBM': LGBMClassifier(random_state=42),
    'Stochastic Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Sonuçları saklamak için sözlük
results = {}

# Her bir modeli eğit ve test et
for name, model in models.items():
    print(f"\n{name} modeli eğitiliyor...")

    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahminler yap
    y_pred = model.predict(X_test)

    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    # Sonuçları sakla
    results[name] = {
        'accuracy': accuracy,
        'report': report
    }

    print(f"\n{name} Sonuçları:")
    print(f"Doğruluk: {accuracy:.4f}")
    print("\nDetaylı Sınıflandırma Raporu:")
    print(report)
    print("-" * 80)

# En iyi modeli bul
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nEn iyi performans gösteren model: {best_model[0]} (Doğruluk: {best_model[1]['accuracy']:.4f})")