from layer_definitions import Dense, Tanh, mse, mse_prime
import numpy as np

# Defines the training dataset.
X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Lays out the structure for the network.
neural_net = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# Number of iterations to train
epochs = 10000

# How fast to train
learning_rate = .1

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):

        # Output of one layer becomes the input of another.
        output = x
        for layer in neural_net:
            output = layer.forward(output)
        
        # Calculates the MSE (for display purposes)
        error = mse(y, output)

        # Calculates the derivative of MSE for gradient descent.
        grad = mse_prime(y, output)

        # Calls the backward learning for each layer.
        for layer in reversed(neural_net):
            grad = layer.backward(grad, learning_rate)
        
        error /= len(X)

# Test one instance of XOR.
output = [[0], [1]]
for layer in neural_net:
    output = layer.forward(output)
    print(output)
