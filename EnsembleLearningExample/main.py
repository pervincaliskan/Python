import os
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio


class SARImageClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.image_size = (128, 128)

    def load_and_preprocess(self, base_path):
        """Sentinel-1 SAR görüntülerini yükle ve ön işle"""
        img_path = os.path.join(base_path, 'img_dir')
        ann_path = os.path.join(base_path, 'ann_dir')

        images = []
        labels = []
        all_means = []  # Önce tüm ortalama değerleri toplayalım

        # İlk geçiş - tüm ortalama değerleri topla
        img_files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
        for img_file in img_files:
            ann_filepath = os.path.join(ann_path, img_file)
            if os.path.exists(ann_filepath):
                with rasterio.open(ann_filepath) as src:
                    label = src.read(1)
                    all_means.append(np.mean(label))

        # Kuantilleri hesapla
        q33, q66 = np.percentile(all_means, [33, 66])
        print(f"Eşik değerleri - Alt: {q33:.3f}, Üst: {q66:.3f}")

        # İkinci geçiş - görüntüleri işle ve etiketleri ata
        for img_file in img_files:
            img_filepath = os.path.join(img_path, img_file)
            ann_filepath = os.path.join(ann_path, img_file)

            with rasterio.open(img_filepath) as src:
                img = src.read(1)
                img = tf.image.resize(
                    tf.expand_dims(img, axis=-1),
                    self.image_size
                ).numpy()
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

            if os.path.exists(ann_filepath):
                with rasterio.open(ann_filepath) as src:
                    label = src.read(1)
                    label_mean = np.mean(label)

                    if label_mean <= q33:
                        label_class = 0
                    elif label_mean <= q66:
                        label_class = 1
                    else:
                        label_class = 2

            images.append(img)
            labels.append(label_class)

        X = np.array(images)
        y = np.array(labels)

        # Sınıf dağılımını göster
        unique, counts = np.unique(y, return_counts=True)
        print("\nSınıf dağılımı:")
        for label, count in zip(unique, counts):
            print(f"Sınıf {label}: {count} örnek")

        return X, y

    def extract_features(self, X):
        """Geliştirilmiş özellik çıkarımı"""
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.image_size, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5)
        ])

        feature_extractor.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        features = feature_extractor.predict(X, verbose=0)
        return features

    def build_models(self, y_train):
        """Geliştirilmiş model oluşturma"""
        # Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }
        rf_model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            n_jobs=-1
        )

        # Bagging - daha basit parametreler ile
        bagging_params = {
            'n_estimators': [10, 20],
            'max_samples': [0.5, 1.0]
        }
        bagging_model = GridSearchCV(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                random_state=42
            ),
            bagging_params,
            cv=3,
            n_jobs=-1
        )

        # Modelleri dictionary'e ekle
        self.models['random_forest'] = rf_model
        self.models['bagging'] = bagging_model

        return self.models

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Tüm modelleri eğit ve değerlendir"""
        results = {}
        X_train_features = self.extract_features(X_train)
        X_test_features = self.extract_features(X_test)

        # Modelleri oluştur
        self.build_models(y_train)

        # Her model için eğitim ve değerlendirme yap
        for name, model in self.models.items():
            print(f"\n{name} eğitiliyor...")
            # Model eğitimi
            model.fit(X_train_features, y_train)

            # Tahminler
            y_pred = model.predict(X_test_features)

            # Performans metrikleri
            accuracy = np.mean(y_pred == y_test)
            cm = tf.math.confusion_matrix(y_test, y_pred)

            # Sonuçları kaydet
            results[name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'best_params': model.best_params_ if hasattr(model, 'best_params_') else None
            }

        return results

    def visualize_results(self, results, class_names):
        """Tüm modellerin sonuçlarını görselleştir"""
        for name, result in results.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(result['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.title(f'{name} - SAR Görüntü Sınıflandırma Confusion Matrix')
            plt.xlabel('Tahmin Edilen Sınıf')
            plt.ylabel('Gerçek Sınıf')
            plt.show()

            print(f"\n{name} Sonuçları:")
            print(f"Doğruluk: {result['accuracy']:.4f}")
            if result['best_params']:
                print("En iyi parametreler:", result['best_params'])


def main():
    classifier = SARImageClassifier()
    X, y = classifier.load_and_preprocess('Turkey')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    class_names = ['Düşük', 'Orta', 'Yüksek']
    classifier.visualize_results(results, class_names)


if __name__ == "__main__":
    main()