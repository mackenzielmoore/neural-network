import numpy as np

np.random.seed(0)


# Dense layer neurons
# It's dense as the neurons of the layer are connected to every neuron of its preceding layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Rectified Linear Unit activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Converts raw scores (logits) into probabilities
class Activation_Softmax:
    def forward(self, inputs):
        # Subtract the maximum value of each set of inputs from the inputs themselves to prevent overflow.
        # Exponentiate each stabilised input value to ensure all values are positive.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalise exponentiated values by dividing by the sum of exponentials for each set of inputs,
        # producing a probability distribution.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


def build_model(X):
    # n_inputs must be 2, as it uses coordinate data (x, y)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # n_inputs must be 3, as the previous layer has 3 neurons
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Print output
    print(activation2.output[:5])
