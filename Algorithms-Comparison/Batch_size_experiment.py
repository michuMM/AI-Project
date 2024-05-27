import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Załadowanie danych
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Przekształcenie danych 2D na 3D (liczba próbek, liczba kroków czasowych, liczba cech)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def build_and_train_model(batch_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    # Kompilacja modelu
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Trenowanie modelu
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    # Predykcja na danych testowych
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Obliczenie dokładności
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    return accuracy

def run_experiment_batch_size():
    batch_sizes = [16, 32, 64, 128, 256]
    accuracies = []

    for batch_size in batch_sizes:
        accuracy = build_and_train_model(batch_size)
        accuracies.append(accuracy)
        print(f'Batch size: {batch_size}, Accuracy: {accuracy}')

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, accuracies, marker='o')
    plt.title('Dokładność modelu względem rozmiaru batch size')
    plt.xlabel('Rozmiar batch size')
    plt.ylabel('Dokładność modelu (%)')
    plt.grid(True)
    plt.show()

run_experiment_batch_size()
