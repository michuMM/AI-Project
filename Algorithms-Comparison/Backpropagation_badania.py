import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Załadowanie danych
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

# Normalizacja danych
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Funkcja trenująca model
def train_model(hidden_layer1_neurons, hidden_layer2_neurons, learning_rate, epochs=1000):
    input_layer_neurons = X_train.shape[1]
    output_layer_neurons = y_train.shape[1]

    # Inicjalizacja wag
    wh1 = np.random.uniform(size=(input_layer_neurons, hidden_layer1_neurons))
    bh1 = np.random.uniform(size=(1, hidden_layer1_neurons))
    wh2 = np.random.uniform(size=(hidden_layer1_neurons, hidden_layer2_neurons))
    bh2 = np.random.uniform(size=(1, hidden_layer2_neurons))
    wo = np.random.uniform(size=(hidden_layer2_neurons, output_layer_neurons))
    bo = np.random.uniform(size=(1, output_layer_neurons))

    for epoch in range(epochs):
        # Forward Propagation
        hidden_layer1_input = np.dot(X_train, wh1) + bh1
        hidden_layer1_output = sigmoid(hidden_layer1_input)

        hidden_layer2_input = np.dot(hidden_layer1_output, wh2) + bh2
        hidden_layer2_output = sigmoid(hidden_layer2_input)

        output_layer_input = np.dot(hidden_layer2_output, wo) + bo
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y_train - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer2 = d_predicted_output.dot(wo.T)
        d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer2_output)

        error_hidden_layer1 = d_hidden_layer2.dot(wh2.T)
        d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer1_output)

        # Aktualizacja wag i biasów
        wo += hidden_layer2_output.T.dot(d_predicted_output) * learning_rate
        bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

        wh2 += hidden_layer1_output.T.dot(d_hidden_layer2) * learning_rate
        bh2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * learning_rate

        wh1 += X_train.T.dot(d_hidden_layer1) * learning_rate
        bh1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * learning_rate

    # Obliczanie dokładności
    hidden_layer1_input = np.dot(X_test, wh1) + bh1
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, wh2) + bh2
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, wo) + bo
    predicted_output = sigmoid(output_layer_input)

    predicted_output = np.round(predicted_output)
    accuracy = np.mean(predicted_output == y_test) * 100
    return accuracy
# Dokładaność zależności od liczby neuronów
# # Generowanie większej tablicy neuronów
# neuron_counts_1 = np.linspace(16, 80, 10, dtype=int)
# neuron_counts_2 = np.linspace(16, 80, 10, dtype=int)
#
#
# # Eksperyment 1: Liczba neuronów w warstwach ukrytych
# results_neurons = []
#
# for neurons_1 in neuron_counts_1:
#     for neurons_2 in neuron_counts_2:
#         accuracy = train_model(neurons_1, neurons_2, learning_rate=0.01, epochs=2000)
#         results_neurons.append((neurons_1, neurons_2, accuracy))
#
# # Przetwarzanie wyników
# neuron_counts_1, neuron_counts_2, accuracies_neurons = zip(*results_neurons)
# neuron_counts_1 = np.array(neuron_counts_1)
# neuron_counts_2 = np.array(neuron_counts_2)
# accuracies_neurons = np.array(accuracies_neurons)
#
# # Generowanie wykresu 3D
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Tworzenie wykresu 3D
# ax.plot_trisurf(neuron_counts_1, neuron_counts_2, accuracies_neurons, cmap='viridis')
#
# ax.set_title('Dokładność zależności od liczby neuronów w warstwach ukrytych')
# ax.set_xlabel('Neurony w pierwszej warstwie ukrytej')
# ax.set_ylabel('Neurony w drugiej warstwie ukrytej')
# ax.set_zlabel('Dokładność modelu (%)')
#
# plt.show()

# Dodkładność zależności od współczynnika uczenia
learning_rates = np.logspace(-4, 0, 30)
results_learning_rates = []

for lr in learning_rates:
    accuracy = train_model(hidden_layer1_neurons=64, hidden_layer2_neurons=32, learning_rate=lr, epochs=1000)
    results_learning_rates.append((lr, accuracy))

# Przetwarzanie wyników
learning_rates, accuracies_lr = zip(*results_learning_rates)
learning_rates = np.array(learning_rates)
accuracies_lr = np.array(accuracies_lr)

plt.figure(figsize=(10, 6))
plt.plot(learning_rates, accuracies_lr, marker='o')
plt.title('Dokładność zależności od współczynnika uczenia')
plt.xlabel('Współczynnik uczenia')
plt.ylabel('Dokładność modelu (%)')
plt.xscale('log')
plt.grid(True)
plt.show()
