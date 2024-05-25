import numpy as np
import pandas as pd

# Załadowanie danych
X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values
y_test = pd.read_csv('y_test.csv').values

#Normalizacja danych
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_layer_neurons = X_train.shape[1]
hidden_layer1_neurons = 64
hidden_layer2_neurons = 32
output_layer_neurons = y_train.shape[1]

# Inicjalizacja wag
wh1 = np.random.uniform(size=(input_layer_neurons, hidden_layer1_neurons))
bh1 = np.random.uniform(size=(1, hidden_layer1_neurons))
wh2 = np.random.uniform(size=(hidden_layer1_neurons, hidden_layer2_neurons))
bh2 = np.random.uniform(size=(1, hidden_layer2_neurons))
wo = np.random.uniform(size=(hidden_layer2_neurons, output_layer_neurons))
bo = np.random.uniform(size=(1, output_layer_neurons))

learning_rate = 0.01
epochs = 1000

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

    if epoch % 100 == 0:
        loss = np.mean(np.square(y_train - predicted_output))
        print(f'Epoch {epoch}, Loss: {loss}')
