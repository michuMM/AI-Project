import numpy as np
import pandas as pd

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

# Hyperparametry Adam
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 1000

# Inicjalizacja parametrów Adam
m_wh1, v_wh1 = np.zeros_like(wh1), np.zeros_like(wh1)
m_bh1, v_bh1 = np.zeros_like(bh1), np.zeros_like(bh1)
m_wh2, v_wh2 = np.zeros_like(wh2), np.zeros_like(wh2)
m_bh2, v_bh2 = np.zeros_like(bh2), np.zeros_like(bh2)
m_wo, v_wo = np.zeros_like(wo), np.zeros_like(wo)
m_bo, v_bo = np.zeros_like(bo), np.zeros_like(bo)

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

    # Aktualizacja wag i biasów z użyciem Adam
    t = epoch + 1

    # Aktualizacja wo i bo
    m_wo = beta1 * m_wo + (1 - beta1) * hidden_layer2_output.T.dot(d_predicted_output)
    v_wo = beta2 * v_wo + (1 - beta2) * (hidden_layer2_output.T.dot(d_predicted_output) ** 2)
    m_wo_hat = m_wo / (1 - beta1 ** t)
    v_wo_hat = v_wo / (1 - beta2 ** t)
    wo += learning_rate * m_wo_hat / (np.sqrt(v_wo_hat) + epsilon)

    m_bo = beta1 * m_bo + (1 - beta1) * np.sum(d_predicted_output, axis=0, keepdims=True)
    v_bo = beta2 * v_bo + (1 - beta2) * (np.sum(d_predicted_output, axis=0, keepdims=True) ** 2)
    m_bo_hat = m_bo / (1 - beta1 ** t)
    v_bo_hat = v_bo / (1 - beta2 ** t)
    bo += learning_rate * m_bo_hat / (np.sqrt(v_bo_hat) + epsilon)

    # Aktualizacja wh2 i bh2
    m_wh2 = beta1 * m_wh2 + (1 - beta1) * hidden_layer1_output.T.dot(d_hidden_layer2)
    v_wh2 = beta2 * v_wh2 + (1 - beta2) * (hidden_layer1_output.T.dot(d_hidden_layer2) ** 2)
    m_wh2_hat = m_wh2 / (1 - beta1 ** t)
    v_wh2_hat = v_wh2 / (1 - beta2 ** t)
    wh2 += learning_rate * m_wh2_hat / (np.sqrt(v_wh2_hat) + epsilon)

    m_bh2 = beta1 * m_bh2 + (1 - beta1) * np.sum(d_hidden_layer2, axis=0, keepdims=True)
    v_bh2 = beta2 * v_bh2 + (1 - beta2) * (np.sum(d_hidden_layer2, axis=0, keepdims=True) ** 2)
    m_bh2_hat = m_bh2 / (1 - beta1 ** t)
    v_bh2_hat = v_bh2 / (1 - beta2 ** t)
    bh2 += learning_rate * m_bh2_hat / (np.sqrt(v_bh2_hat) + epsilon)

    # Aktualizacja wh1 i bh1
    m_wh1 = beta1 * m_wh1 + (1 - beta1) * X_train.T.dot(d_hidden_layer1)
    v_wh1 = beta2 * v_wh1 + (1 - beta2) * (X_train.T.dot(d_hidden_layer1) ** 2)
    m_wh1_hat = m_wh1 / (1 - beta1 ** t)
    v_wh1_hat = v_wh1 / (1 - beta2 ** t)
    wh1 += learning_rate * m_wh1_hat / (np.sqrt(v_wh1_hat) + epsilon)

    m_bh1 = beta1 * m_bh1 + (1 - beta1) * np.sum(d_hidden_layer1, axis=0, keepdims=True)
    v_bh1 = beta2 * v_bh1 + (1 - beta2) * (np.sum(d_hidden_layer1, axis=0, keepdims=True) ** 2)
    m_bh1_hat = m_bh1 / (1 - beta1 ** t)
    v_bh1_hat = v_bh1 / (1 - beta2 ** t)
    bh1 += learning_rate * m_bh1_hat / (np.sqrt(v_bh1_hat) + epsilon)

    if epoch % 100 == 0:
        loss = np.mean(np.square(y_train - predicted_output))
        print(f'Epoch {epoch}, Loss: {loss}')
