import numpy as np

np.random.seed(0)


# Each neuron in this layer is connected to every neuron in the previous layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values and biases with zeros
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Compute the output of the layer as a weighted sum of inputs plus biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Rectified Linear Unit (ReLU) activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Replaces negative values with 0
        self.output = np.maximum(0, inputs)


# Softmax activation function
# Converts raw predictions to probabilities by normalising exponentiated values
class Activation_Softmax:
    def forward(self, inputs):
        # For numerical stability, subtract the max value from inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize by dividing by the sum of exponentials to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Base class for loss functions
# Calculates the loss given the output and true labels
class Loss:
    def calculate(self, output, y):
        # Compute the loss for each sample and average it
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss


# Categorical Cross-Entropy Loss
# Measures the performance of a classification model whose output is probabilities
# y_pred is the predicted target class
# y_true is the expected target class
class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip predictions to prevent log(0) and numerical instability
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If true labels are in integer form, select the predicted probabilities of the true class
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # If true labels are one-hot encoded, sum the predicted probabilities of the true classes
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Compute the negative log likelihood of the correct classes
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def build_model(X, y):
    # Initialize layers
    # Input layer: 2 features -> 3 neurons
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    # Hidden layer: 3 neurons -> 3 neurons
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    # Forward pass through the network
    dense1.forward(X)  # Compute outputs for the first dense layer
    activation1.forward(dense1.output)  # Apply ReLU activation

    dense2.forward(activation1.output)  # Compute outputs for the second dense layer
    activation2.forward(dense2.output)  # Apply softmax activation

    # Output the softmax probabilities for the first 5 samples
    print(activation2.output[:5])

    # Compute the loss
    loss_function = Loss_Categorical_Cross_Entropy()
    loss = loss_function.calculate(activation2.output, y)

    # Print the loss value
    print("Loss:", loss)


# Next steps
# Training Loop:
# Implement a training loop to adjust weights and biases based on the loss.
# Use an optimization algorithm like gradient descent.

# Backward Propagation:
# Add the backward pass to compute gradients of weights and biases using the chain rule. This is necessary for updating the weights during training.

#  -  Implement weight updates to minimize the loss, often using gradient descent or its variants.
#  -  Evaluate the model's performance on unseen data.
