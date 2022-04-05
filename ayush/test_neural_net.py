import numpy as np

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