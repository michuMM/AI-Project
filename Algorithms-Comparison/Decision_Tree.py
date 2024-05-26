import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# Obliczenie dokładności
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
