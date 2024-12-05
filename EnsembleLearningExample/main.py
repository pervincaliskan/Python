
import os
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio


class RotationForestClassifier:
    def __init__(self, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        np.random.seed(random_state)
        self.trees = []
        self.pcas = []
        self.oob_score_ = 0  # OOB skoru için

    def fit(self, X, y, sample_weight=None):
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, self.n_estimators))
        n_times_used = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # Bootstrap örnekleme
            bootstrap_indices = np.random.choice(n_samples, size=int(0.67 * n_samples), replace=True)
            oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))

            # InBag verilerini kullan
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # PCA uygula
            pca = PCA(random_state=self.random_state)
            X_transformed = pca.fit_transform(X_bootstrap)

            # Ağaç eğit
            tree = DecisionTreeClassifier(random_state=self.random_state)
            if sample_weight is not None:
                tree.fit(X_transformed, y_bootstrap, sample_weight=sample_weight[bootstrap_indices])
            else:
                tree.fit(X_transformed, y_bootstrap)

            # OOB tahminleri
            if len(oob_indices) > 0:
                X_oob = X[oob_indices]
                X_oob_transformed = pca.transform(X_oob)
                oob_predictions[oob_indices, i] = tree.predict(X_oob_transformed)
                n_times_used[oob_indices] += 1

            self.trees.append(tree)
            self.pcas.append(pca)

        # OOB skoru hesapla
        oob_predictions_mean = []
        for i in range(n_samples):
            if n_times_used[i] > 0:
                predictions = oob_predictions[i, :int(n_times_used[i])]
                oob_predictions_mean.append(np.bincount(predictions.astype(int)).argmax())

        self.oob_score_ = accuracy_score(y[n_times_used > 0], oob_predictions_mean)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for idx, (tree, pca) in enumerate(zip(self.trees, self.pcas)):
            X_transformed = pca.transform(X)
            predictions[:, idx] = tree.predict(X_transformed)

        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=1,
            arr=predictions
        )


class SARImageClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.image_size = (64, 64)

    def load_and_preprocess(self, base_path):
        """Sentinel-1 SAR görüntülerini yükle ve ön işle"""
        img_path = os.path.join(base_path, 'img_dir')
        ann_path = os.path.join(base_path, 'ann_dir')

        images = []
        labels = []
        all_means = []

        img_files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
        for img_file in img_files:
            ann_filepath = os.path.join(ann_path, img_file)
            if os.path.exists(ann_filepath):
                with rasterio.open(ann_filepath) as src:
                    label = src.read(1)
                    all_means.append(np.mean(label))

        q33, q66 = np.percentile(all_means, [33, 66])
        print(f"Eşik değerleri - Alt: {q33:.3f}, Üst: {q66:.3f}")

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

        return X, y

    def extract_features(self, X):
        """SAR görüntülerinden özellik çıkarımı"""
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.image_size, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])

        feature_extractor.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        features = feature_extractor.predict(X, batch_size=64, verbose=0)
        return features

    def build_models(self, X, y):
        """Tüm ensemble modellerini oluştur"""
        # Sınıf ağırlıklarını hesapla
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y
        )

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight='balanced',
            oob_score=True,
            random_state=42
        )

        # Bagging
        bagging_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight='balanced'),
            n_estimators=20,
            max_samples=0.67,  # InBag oranı
            oob_score=True,
            random_state=42
        )

        # Rotation Forest
        rotation_model = RotationForestClassifier(n_estimators=20, random_state=42)

        # Voting Classifier
        voting_model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
                ('bag', BaggingClassifier(n_estimators=20, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
            ],
            voting='soft'
        )

        # Stacking Classifier
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('bag', BaggingClassifier(n_estimators=20, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            cv=5
        )

        # Modelleri dictionary'e ekle
        self.models['random_forest'] = rf_model
        self.models['bagging'] = bagging_model
        self.models['rotation_forest'] = rotation_model
        self.models['voting'] = voting_model
        self.models['stacking'] = stacking_model

        return self.models

    def train_and_evaluate(self, X, y):
        """Tüm modelleri eğit ve değerlendir"""
        results = {}
        features = self.extract_features(X)

        # Veriyi InBag ve OOB olarak böl
        n_samples = len(y)
        inbag_indices = np.random.choice(n_samples, size=int(0.67 * n_samples), replace=False)
        oob_indices = list(set(range(n_samples)) - set(inbag_indices))

        X_inbag = features[inbag_indices]
        y_inbag = y[inbag_indices]
        X_oob = features[oob_indices]
        y_oob = y[oob_indices]

        # Sınıf ağırlıklarını hesapla
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_inbag
        )

        self.build_models(X_inbag, y_inbag)

        for name, model in self.models.items():
            print(f"\n{name} eğitiliyor...")
            model.fit(X_inbag, y_inbag, sample_weight=sample_weights)

            # OOB verileriyle değerlendir
            y_pred = model.predict(X_oob)
            accuracy = accuracy_score(y_oob, y_pred)
            cm = confusion_matrix(y_oob, y_pred)

            # OOB skoru
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None

            results[name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'oob_score': oob_score
            }

        return results

    def visualize_results(self, results, class_names):
        """Tüm modellerin sonuçlarını görselleştir"""
        for name, result in results.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title(f'{name} - SAR Görüntü Sınıflandırma Confusion Matrix')
            plt.xlabel('Tahmin Edilen Sınıf')
            plt.ylabel('Gerçek Sınıf')
            plt.tight_layout()
            plt.show()

            print(f"\n{name} Sonuçları:")
            print(f"Test Doğruluğu: {result['accuracy']:.4f}")
            if result['oob_score'] is not None:
                print(f"OOB Skoru: {result['oob_score']:.4f}")

        plt.figure(figsize=(10, 6))
        accuracies = {name: result['accuracy'] for name, result in results.items()}
        bars = plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Doğruluk Karşılaştırması')
        plt.xlabel('Model')
        plt.ylabel('Doğruluk')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def main():
    classifier = SARImageClassifier()
    X, y = classifier.load_and_preprocess('Turkey')
    results = classifier.train_and_evaluate(X, y)
    class_names = ['Düşük', 'Orta', 'Yüksek']
    classifier.visualize_results(results, class_names)


if __name__ == "__main__":
    main()
