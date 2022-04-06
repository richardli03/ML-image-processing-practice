from layer_definitions import Convolutional, Sigmoid, Reshape, Dense, binary_cross_entropy_prime
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

## MNIST 0/1 test
def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    #x, y = x[all_indices], y[all_indices]
    x = x.reshape(x.shape[0], 1, 28, 28)
    x = x.astype("float32")/255
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 2000)
x_test, y_test = preprocess_data(x_test, y_test, 200)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5*26*26, 1)),
    Dense(5*26*26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

epochs = 60

learning_rate = .1

for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        output = x
        for layer in network:
            output = layer.forward(output)
        pred = np.argmax(output)
        true = np.argmax(y)
        if pred == true:
            error += 1
        grad = binary_cross_entropy_prime(y, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    print(f"{e}, {error/len(x_train)}")
# Test
error = 0
for x, y in zip(x_test[:-1], y_test[:-1]):
    output = x
    for layer in network:
        output = layer.forward(output)
    if np.argmax(output) == np.argmax(y):
        error += 1
print(f"final error: {error/len(x_test)}")

plt.imshow(np.reshape(x_test[-1], (28, 28)), cmap='gray')

output = x_test[-1]
for layer in network:
    output = layer.forward(output)
print(f"This is number is predicted to be: {np.argmax(output)}")
output = np.delete(output, np.argmax(output))
print(f"Second Guess: {np.argmax(output)}")
print(f"This number actually is: {np.argmax(y_test[-1])}")
plt.show()