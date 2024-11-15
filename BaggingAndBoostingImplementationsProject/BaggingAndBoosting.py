import numpy as np


def spor_veri_seti_olustur():
    np.random.seed(42)

    secilen_boy = np.random.normal(180, 5, 50)
    secilen_kilo = np.random.normal(75, 8, 50)
    secilen_yas = np.random.normal(16, 1, 50)

    secilmeyen_boy = np.random.normal(170, 5, 50)
    secilmeyen_kilo = np.random.normal(65, 8, 50)
    secilmeyen_yas = np.random.normal(15, 1, 50)

    X = np.vstack([
        np.column_stack((secilen_boy, secilen_kilo, secilen_yas)),
        np.column_stack((secilmeyen_boy, secilmeyen_kilo, secilmeyen_yas))
    ])

    y = np.hstack([np.ones(50), np.zeros(50)])

    return X, y


def veri_seti_bol(X, y, test_orani=0.2):
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    test_size = int(len(X) * test_orani)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return (X[train_indices], X[test_indices],
            y[train_indices], y[test_indices])


class KararAgaci:
    def __init__(self, max_derinlik=3):
        self.max_derinlik = max_derinlik

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        self.boy_esik = np.average(X[:, 0], weights=sample_weight)
        self.kilo_esik = np.average(X[:, 1], weights=sample_weight)
        self.yas_esik = np.average(X[:, 2], weights=sample_weight)

    def predict(self, X):
        tahminler = []
        for x in X:
            tahmin = 1 if (x[0] > self.boy_esik and
                           x[1] > self.kilo_esik and
                           x[2] > self.yas_esik) else 0
            tahminler.append(tahmin)
        return np.array(tahminler)


class Bagging:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            model = KararAgaci()
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        tahminler = []
        for model in self.models:
            tahmin = model.predict(X)
            tahminler.append(tahmin)

        tahminler = np.array(tahminler)
        final_tahminler = []
        for i in range(len(X)):
            oy_1 = sum(tahminler[:, i] == 1)
            oy_0 = sum(tahminler[:, i] == 0)
            final_tahminler.append(1 if oy_1 > oy_0 else 0)

        return np.array(final_tahminler)


class AdaBoost:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        sample_weights = np.ones(len(y)) / len(y)

        for _ in range(self.n_estimators):
            model = KararAgaci()
            model.fit(X, y, sample_weights)

            predictions = model.predict(X)

            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            model_weight = np.log((1 - error) / max(error, 1e-10))

            sample_weights *= np.exp(model_weight * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.models.append(model)
            self.model_weights.append(model_weight)

    def predict(self, X):
        tahminler = np.zeros(len(X))
        for model, weight in zip(self.models, self.model_weights):
            tahminler += weight * model.predict(X)

        return (tahminler > 0).astype(int)


if __name__ == "__main__":
    print("Veri seti oluşturuluyor...")
    X, y = spor_veri_seti_olustur()
    print(f"Veri seti boyutu: {len(X)} öğrenci")
    print("\nİlk 5 öğrenci bilgileri:")
    for i in range(5):
        print(f"Öğrenci {i + 1}: Boy={X[i, 0]:.1f}cm, Kilo={X[i, 1]:.1f}kg, " +
              f"Yaş={X[i, 2]:.1f}, Seçildi mi? {'Evet' if y[i] == 1 else 'Hayır'}")

    print("\nVeri eğitim ve test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = veri_seti_bol(X, y)
    print(f"Eğitim seti: {len(X_train)} öğrenci")
    print(f"Test seti: {len(X_test)} öğrenci")

    print("\nBagging modeli eğitiliyor...")
    bagging = Bagging(n_estimators=5)
    bagging.fit(X_train, y_train)
    bagging_tahminler = bagging.predict(X_test)
    bagging_basari = np.mean(bagging_tahminler == y_test)
    print(f"Bagging başarı oranı: {bagging_basari:.2%}")

    print("\nAdaBoost modeli eğitiliyor...")
    boosting = AdaBoost(n_estimators=5)
    boosting.fit(X_train, y_train)
    boosting_tahminler = boosting.predict(X_test)
    boosting_basari = np.mean(boosting_tahminler == y_test)
    print(f"AdaBoost başarı oranı: {boosting_basari:.2%}")

    print("\nÖrnek tahminler:")
    ornek_ogrenciler = [
        [185, 80, 17],
        [165, 60, 14],
        [175, 70, 16]
    ]

    print("\nBagging tahminleri:")
    for i, ogrenci in enumerate(ornek_ogrenciler):
        tahmin = bagging.predict([ogrenci])[0]
        print(f"Öğrenci {i + 1} (Boy={ogrenci[0]}cm, Kilo={ogrenci[1]}kg, " +
              f"Yaş={ogrenci[2]}): {'Seçildi' if tahmin == 1 else 'Seçilmedi'}")

    print("\nAdaBoost tahminleri:")
    for i, ogrenci in enumerate(ornek_ogrenciler):
        tahmin = boosting.predict([ogrenci])[0]
        print(f"Öğrenci {i + 1} (Boy={ogrenci[0]}cm, Kilo={ogrenci[1]}kg, " +
              f"Yaş={ogrenci[2]}): {'Seçildi' if tahmin == 1 else 'Seçilmedi'}")