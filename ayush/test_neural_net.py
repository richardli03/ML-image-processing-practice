import numpy as np
from scipy import signal
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt


class Layer:
    """
    Basic Layer definition that allows for general layer use in runner for loop.
    Has three main methods, constructor that specifies the input and output size
    of the layer, forward direction to generate an output, and backward direction
    that implements gradient descent to tweak weights and biases.
    """
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    """
    Dense Layer implementation which is basically just a fully connected layer.
    i.e. one neuron in this layer is connected to every single neuron in the previous
    layer. basically the output of each neuron is calculated through y1 = w1(x1) + w2(x2) + ... wi(xi)
    for every neuron on the output where y1 represents the first outputted neuron and x1 represents
    the first inputted neuron.
    """
    def __init__(self, input_size, output_size):
        # Randomly assigns initial weights
        self.weights = np.random.randn(output_size, input_size)
        # Randomly assigns initial biases
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        # In the forward direction, the ouput is simply the dot product of the weights
        # matrices (j x ji size) by the input matrices (i x 1 size) to give the output
        # and the biases are added to the result 
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # For the backwards direction, the weights and biases need to be tweaked,
        # along with the generation of another par E par Y to input into the previous
        # layer's learning. For this, the weights gradient is just the output gradient
        # times the transpose of X (derived in video) and the change of the biases is just
        # equal to par E par Y (derived in video). To protect against overcorrection or that
        # stuff, there's the learning rate.
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate*weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Activation(Layer):
    # This is an activation layer, which basically just turns the linear operations of the
    # fully connected layer into more non-linear operations to ensure that no overfitting to
    # specific trends in the training data is seen. For this, the hyperbolic tan curve was
    # used. 
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        # Forward direction just applies the activation function on everything in the input.
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Backward direction just applies the derivative of the activation function on the 
        # par E par Y. 
         return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    # This is a specific activation function that is a subclass of Activation.
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'valid')
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], 'full')
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s*(1-s)
        super().__init__(sigmoid, sigmoid_prime)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1-y_true)/(1-y_pred)-y_true/y_pred)/np.size(y_true)

def mse(y_true, y_pred):
    """
    This computes the mean standard error between the predicted and actual values.
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """
    This computes the derivative of the mean standard error between the predicted and
    actual values from the specific epoch.
    """
    return 2*(y_pred-y_true)/np.size(y_true)

## XOR Test
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