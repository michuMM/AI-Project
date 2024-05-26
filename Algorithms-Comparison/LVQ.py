import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class LVQ:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.prototypes = None
        self.prototype_labels = None

    def _initialize_prototypes(self, X, y):
        n_classes = len(np.unique(y))
        self.prototypes = np.zeros((n_classes * self.n_prototypes_per_class, X.shape[1]))
        self.prototype_labels = np.zeros(n_classes * self.n_prototypes_per_class)

        for i in range(n_classes):
            class_indices = np.where(y == i)[0]
            selected_indices = np.random.choice(class_indices, self.n_prototypes_per_class, replace=False)
            self.prototypes[i * self.n_prototypes_per_class:(i + 1) * self.n_prototypes_per_class] = X[selected_indices]
            self.prototype_labels[i * self.n_prototypes_per_class:(i + 1) * self.n_prototypes_per_class] = y[
                selected_indices]

    def _update_prototype(self, x, y):
        distances = np.linalg.norm(self.prototypes - x, axis=1)
        winner_index = np.argmin(distances)
        winner_label = self.prototype_labels[winner_index]

        if winner_label == y:
            self.prototypes[winner_index] += self.learning_rate * (x - self.prototypes[winner_index])
        else:
            self.prototypes[winner_index] -= self.learning_rate * (x - self.prototypes[winner_index])

    def fit(self, X, y):
        self._initialize_prototypes(X, y)
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                self._update_prototype(X[i], y[i])

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = np.linalg.norm(self.prototypes - X[i], axis=1)
            winner_index = np.argmin(distances)
            y_pred[i] = self.prototype_labels[winner_index]
        return y_pred


# Wczytaj dane z plików CSV
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konwersja one-hot encoded etykiet do jednowymiarowych etykiet
y_train_1d = np.argmax(y_train, axis=1)
y_test_1d = np.argmax(y_test, axis=1)

# Inicjalizacja i trenowanie modelu LVQ
lvq = LVQ(n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100)
lvq.fit(X_train_scaled, y_train_1d)

# Predykcja na danych testowych
y_pred = lvq.predict(X_test_scaled)

# Ewaluacja modelu
accuracy = accuracy_score(y_test_1d, y_pred)
print(f'Accuracy: {accuracy}')
