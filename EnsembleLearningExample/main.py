import os
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio


class SARImageClassifier:
    def __init__(self):
        self.models = {}
        self.features = {}
        self.scaler = StandardScaler()
        self.image_size = (128, 128)

    def load_and_preprocess(self, base_path):
        """Sentinel-1 SAR görüntülerini yükle ve ön işle"""
        img_path = os.path.join(base_path, 'img_dir')
        ann_path = os.path.join(base_path, 'ann_dir')

        # Önce tüm etiket değerlerini topla
        img_files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
        all_labels = []

        for img_file in img_files:
            ann_filepath = os.path.join(ann_path, img_file)
            if os.path.exists(ann_filepath):
                with rasterio.open(ann_filepath) as src:
                    label = src.read(1)
                    all_labels.append(np.mean(label))

        # Etiketleri sırala ve üç eşit parçaya böl
        sorted_labels = np.sort(all_labels)
        third = len(sorted_labels) // 3
        threshold1 = sorted_labels[third]
        threshold2 = sorted_labels[2 * third]

        # Şimdi görüntüleri yükle ve sınıflandır
        images = []
        labels = []

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

                    if label_mean <= threshold1:
                        label_class = 0
                    elif label_mean <= threshold2:
                        label_class = 1
                    else:
                        label_class = 2

            images.append(img)
            labels.append(label_class)

        X = np.array(images)
        y = np.array(labels)

        return X, y

    def extract_features(self, X):
        """SAR görüntülerinden özellik çıkarımı"""
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.image_size, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])

        feature_extractor.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        features = feature_extractor.predict(X)
        return features

    def build_ensemble(self, X_train, y_train):
        """Ensemble modeli oluştur"""
        n_classes = len(np.unique(y_train))

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf)
            ],
            voting='soft'
        )

        return ensemble

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Model eğitimi ve değerlendirmesi"""
        X_train_features = self.extract_features(X_train)
        X_test_features = self.extract_features(X_test)

        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()

        ensemble = self.build_ensemble(X_train_features, y_train_flat)
        ensemble.fit(X_train_features, y_train_flat)

        y_pred = ensemble.predict(X_test_features)
        accuracy = np.mean(y_pred == y_test_flat)

        cm = tf.math.confusion_matrix(y_test_flat, y_pred)

        return accuracy, cm

    def visualize_results(self, confusion_matrix, class_names):
        """Sonuçları görselleştir"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('SAR Görüntü Sınıflandırma Confusion Matrix')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.show()


def main():
    classifier = SARImageClassifier()
    X, y = classifier.load_and_preprocess('Turkey')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    accuracy, cm = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    class_names = ['Düşük', 'Orta', 'Yüksek']
    classifier.visualize_results(cm, class_names)
    print(f"Model Doğruluğu: {accuracy:.4f}")


if __name__ == "__main__":
    main()