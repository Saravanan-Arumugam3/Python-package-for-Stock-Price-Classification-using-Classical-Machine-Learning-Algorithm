import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        # input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def relu_derivative(self, Z):
        dZ = np.array(Z, copy=True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ

    def backward(self, X, y, output):
        # output to hidden layer
        self.delta2 = (output - y) * self.sigmoid_derivative(self.z2)
        self.dW2 = np.dot(self.a1.T, self.delta2)
        self.db2 = np.sum(self.delta2, axis=0, keepdims=True)

        # hidden to input layer
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.z1)
        self.dW1 = np.dot(X.T, self.delta1)
        self.db1 = np.sum(self.delta1, axis=0, keepdims=True)

    def optimize(self, learning_rate):
        # update weights and biases
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, output)

            # Gradient descent
            self.optimize(learning_rate)

    def predict(self, X):
        # predict output for input X
        output = self.forward(X)
        predictions = np.round(output)
        return predictions


# Example usage:
# X_train and y_train should be numpy arrays with shapes (num_samples, num_features)


# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size

#         # Initialize weights and biases
#         self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
#         self.b1 = np.zeros((1, self.hidden_size))
#         self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
#         self.b2 = np.zeros((1, self.output_size))

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def relu(self, x):
#         return np.maximum(0, x)

#     def sigmoid_derivative(self, x):
#         return self.sigmoid(x) * (1 - self.sigmoid(x))

#     def relu_derivative(self, x):
#         return np.where(x > 0, 1, 0)

#     def forward(self, X):
#         # Input to hidden layer
#         self.z1 = np.dot(X, self.W1) + self.b1
#         self.a1 = self.relu(self.z1)

#         # Hidden to output layer
#         self.z2 = np.dot(self.a1, self.W2) + self.b2
#         self.a2 = self.sigmoid(self.z2)

#         return self.a2

#     def backward(self, X, y, output):
#         # Output to hidden layer
#         self.delta2 = output - y
#         self.dW2 = np.dot(self.z1.T, self.delta2)
#         self.db2 = np.sum(self.delta2, axis=0, keepdims=True)

#         # Hidden to input layer
#         self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.z1)
#         self.dW1 = np.dot(X.T, self.delta1)
#         self.db1 = np.sum(self.delta1, axis=0)
    
# #     def backward(self, X, y, output):
        
# #         # Output to hidden layer
# #         self.delta2 = (output - y) * self.sigmoid_derivative(self.z2)
# #         self.dW2 = np.dot(self.a1.T, self.delta2)
# #         self.db2 = np.sum(self.delta2, axis=0, keepdims=True)

# #         # Hidden to input layer
# #         #self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.z1)
# #         self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.z1)
# #         #self.delta1 = np.dot(self.W2, self.delta2.T).T * self.relu_derivative(self.z1)
# #         self.dW1 = np.dot(X.T, self.delta1)
# #         self.db1 = np.sum(self.delta1, axis=0)

#     def optimize(self, learning_rate):
#         # Gradient descent
#         self.W1 -= learning_rate * self.dW1
#         self.b1 -= learning_rate * self.db1
#         self.W2 -= learning_rate * self.dW2
#         self.b2 -= learning_rate * self.db2

#     def train(self, X, y, learning_rate=0.01, num_epochs=1000):
#         for epoch in range(num_epochs):
#             # Forward propagation
#             output = self.forward(X)

#             # Backward propagation
#             self.backward(X, y, output)

#             # Gradient descent
#             self.optimize(learning_rate)

#             # Print error
#             if epoch % 100 == 0:
#                 loss = np.mean(np.square(output - y))
#                 print(f"Epoch {epoch}, loss: {loss}")

#     def predict(self, X, threshold=0.5):
#         output = self.forward(X)
#         return np.where(output > threshold, 1, 0)
