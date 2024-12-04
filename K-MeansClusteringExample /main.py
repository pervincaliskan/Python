import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Veri seti oluştur
n_samples = 300
n_clusters = 3
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=42)


def kmeans_with_visualization(X, k, max_iters=100):
    # Rastgele merkezler seç
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    iteration_data = []

    for iter_num in range(max_iters):
        # Mesafeleri hesapla
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # İterasyon verilerini kaydet
        iteration_data.append({
            'centroids': centroids.copy(),
            'labels': labels.copy(),
            'iter_num': iter_num
        })

        # Yeni merkezleri hesapla
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return iteration_data


# Algoritma çalıştır
iterations = kmeans_with_visualization(X, k=n_clusters)

# Her iterasyonu görselleştir
plt.figure(figsize=(15, 5 * ((len(iterations) + 1) // 2)))
for i, iter_data in enumerate(iterations):
    plt.subplot((len(iterations) + 1) // 2, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=iter_data['labels'], cmap='viridis')
    plt.scatter(iter_data['centroids'][:, 0], iter_data['centroids'][:, 1],
                c='red', marker='x', s=200, linewidth=3, label='Merkezler')
    plt.title(f'İterasyon {iter_data["iter_num"]}')
    plt.legend()
plt.tight_layout()
plt.show()

# Son iterasyondaki merkez koordinatları
print("\nSon Küme Merkezleri:")
final_centroids = iterations[-1]['centroids']
for i, centroid in enumerate(final_centroids):
    print(f"Küme {i + 1}: {centroid}")

print(f"\nToplam İterasyon Sayısı: {len(iterations)}")