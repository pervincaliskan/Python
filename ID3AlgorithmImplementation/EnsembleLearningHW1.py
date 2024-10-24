import math


class ID3Algorithm:
    def __init__(self):
        """
        Veri setini ve özellikleri tanımlıyoruz
        Son sütun her zaman karar (Evet/Hayır) sütunudur
        """
        # Veri seti
        self.data = [
            ["Güneşli", "Az", "Evet", "Yeterli", "Evet"],
            ["Yağmurlu", "Çok", "Hayır", "Az", "Hayır"],
            ["Güneşli", "Az", "Evet", "Az", "Evet"],
            ["Bulutlu", "Az", "Evet", "Yeterli", "Evet"],
            ["Yağmurlu", "Az", "Hayır", "Yeterli", "Hayır"],
            ["Güneşli", "Çok", "Evet", "Yeterli", "Hayır"],
            ["Bulutlu", "Çok", "Hayır", "Az", "Hayır"],
            ["Bulutlu", "Normal", "Evet", "Az", "Evet"],
            ["Yağmurlu", "Normal", "Evet", "Yeterli", "Evet"],
            ["Güneşli", "Normal", "Hayır", "Yeterli", "Evet"]
        ]

        # Özellik isimleri
        self.features = ["Hava Durumu", "Yorgunluk", "Arkadaş", "Para"]

    def calculate_entropy(self, data_subset):
        """Entropy hesaplama"""
        total_rows = len(data_subset)
        if total_rows == 0:
            return 0

        # Sınıfların sayısını say (Evet/Hayır)
        class_counts = {"Evet": 0, "Hayır": 0}
        for row in data_subset:
            class_counts[row[-1]] += 1

        # Entropy hesaplama
        entropy = 0
        for count in class_counts.values():
            if count == 0:
                continue
            prob = count / total_rows
            entropy -= prob * math.log2(prob)

        return entropy

    def calculate_information_gain(self, data_subset, feature_index):
        """Bilgi kazancı hesaplama"""
        total_entropy = self.calculate_entropy(data_subset)

        # Özelliğe göre veriyi grupla
        feature_values = {}
        for row in data_subset:
            value = row[feature_index]
            if value not in feature_values:
                feature_values[value] = []
            feature_values[value].append(row)

        # Ağırlıklı entropy hesapla
        weighted_entropy = 0
        total_rows = len(data_subset)
        for subset in feature_values.values():
            prob = len(subset) / total_rows
            weighted_entropy += prob * self.calculate_entropy(subset)

        return total_entropy - weighted_entropy

    def find_best_split(self, data_subset, remaining_features):
        """En iyi özelliği seç"""
        best_gain = -1
        best_feature = None

        for feature_index in remaining_features:
            gain = self.calculate_information_gain(data_subset, feature_index)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index

        return best_feature

    def build_tree(self, data_subset, remaining_features):
        """Ağacı oluştur"""
        # Tüm örnekler aynı sınıfta mı?
        if len(set(row[-1] for row in data_subset)) == 1:
            return {"type": "leaf", "value": data_subset[0][-1]}

        # Özellik kaldı mı?
        if not remaining_features:
            class_counts = {}
            for row in data_subset:
                class_counts[row[-1]] = class_counts.get(row[-1], 0) + 1
            majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
            return {"type": "leaf", "value": majority_class}

        # En iyi özelliği bul
        best_feature = self.find_best_split(data_subset, remaining_features)

        # Ağaç düğümü oluştur
        tree = {
            "type": "node",
            "feature": self.features[best_feature],
            "branches": {}
        }

        # Alt ağaçları oluştur
        feature_values = set(row[best_feature] for row in data_subset)
        for value in feature_values:
            subset = [row for row in data_subset if row[best_feature] == value]
            new_features = remaining_features - {best_feature}
            tree["branches"][value] = self.build_tree(subset, new_features)

        return tree

    def train(self):
        """Modeli eğit"""
        remaining_features = set(range(len(self.features)))
        self.tree = self.build_tree(self.data, remaining_features)
        return self.tree


def print_tree(tree, indent=""):
    """Ağacı görselleştir"""
    if tree["type"] == "leaf":
        print(f"{indent}→ {tree['value']}")
        return

    print(f"{indent}{tree['feature']}")
    for value, subtree in tree["branches"].items():
        print(f"{indent}|-- {value}")
        print_tree(subtree, indent + "|   ")


# Ana çalıştırma kodu
if __name__ == "__main__":
    # Modeli oluştur ve eğit
    model = ID3Algorithm()
    tree = model.train()

    # Ağacı görselleştir
    print("\nOluşturulan Karar Ağacı:")
    print("------------------------")
    print_tree(tree)