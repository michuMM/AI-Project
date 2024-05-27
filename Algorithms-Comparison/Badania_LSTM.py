import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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


# Funkcja do trenowania i ewaluacji modelu LSTM
def train_and_evaluate_lstm(neurons_layer1, neurons_layer2, learning_rate, epochs=20):
    model = Sequential()
    model.add(LSTM(neurons_layer1, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(neurons_layer2))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


# Eksperyment 1: Liczba neuronów w warstwach ukrytych
neuron_counts_1 = np.linspace(16, 128, 10, dtype=int)
neuron_counts_2 = np.linspace(16, 128, 10, dtype=int)
accuracies_neurons = []

for neurons_1 in neuron_counts_1:
    for neurons_2 in neuron_counts_2:
        accuracy = train_and_evaluate_lstm(neurons_layer1=neurons_1, neurons_layer2=neurons_2, learning_rate=0.01,
                                           epochs=20)
        accuracies_neurons.append((neurons_1, neurons_2, accuracy))

# Przetwarzanie wyników
neuron_counts_1, neuron_counts_2, accuracies_neurons = zip(*accuracies_neurons)
neuron_counts_1 = np.array(neuron_counts_1)
neuron_counts_2 = np.array(neuron_counts_2)
accuracies_neurons = np.array(accuracies_neurons)

# Generowanie wykresu 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Tworzenie wykresu 3D
ax.plot_trisurf(neuron_counts_1, neuron_counts_2, accuracies_neurons, cmap='viridis')

ax.set_title('Dokładność zależności od liczby neuronów w warstwach ukrytych')
ax.set_xlabel('Neurony w pierwszej warstwie ukrytej')
ax.set_ylabel('Neurony w drugiej warstwie ukrytej')
ax.set_zlabel('Dokładność modelu (%)')

plt.show()
