import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Załadowanie danych
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

# Normalizacja danych
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Upewnienie się, że y_train i y_test są jednowymiarowe
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# Inicjalizacja modelu drzewa decyzyjnego
tree = DecisionTreeClassifier(max_depth=5)

# Trenowanie modelu
tree.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = tree.predict(X_test)

# max_depths = np.arange(1, 21)
# accuracies_depth = []
#
# for depth in max_depths:
#     tree = DecisionTreeClassifier(max_depth=depth)
#     tree.fit(X_train, y_train)
#     y_pred = tree.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies_depth.append(accuracy)
#
# # Wykres
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(max_depths, accuracies_depth, marker='o')
# plt.title('Dokładność modelu względem maksymalnej głębokości drzewa')
# plt.xlabel('Maksymalna głębokość drzewa')
# plt.ylabel('Dokładność modelu (%)')
# plt.grid(True)
# plt.show()

min_samples_leaves = np.arange(1, 21)
accuracies_leaf = []

for min_samples in min_samples_leaves:
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_leaf.append(accuracy)

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(min_samples_leaves, accuracies_leaf, marker='o')
plt.title('Dokładność modelu względem minimalnej liczby próbek w liściu')
plt.xlabel('Minimalna liczba próbek w liściu')
plt.ylabel('Dokładność modelu (%)')
plt.grid(True)
plt.show()
