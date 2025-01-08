import torch
class XORNet:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.1):
        # Initialize parameters
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=False) # auto derivative is not required
        self.b1 = torch.randn(hidden_size, requires_grad=False)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=False)
        self.b2 = torch.randn(output_size, requires_grad=False)
        self.lr = lr

    
    def sigmoid(self, x):
        # activation function
        return 1 / (1 + torch.exp(-x))
    
    def sigmoid_derivative(self, x):
        # derivative of the activation function
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, X):
        """ Forward pass """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        """ Backward pass (manual gradients computation) """
        dL_da2 = 2 * (self.a2 - y) / y.size(0)       # dL/da2
        da2_dz2 = self.sigmoid_derivative(self.z2)   # da2/dz2
        dL_dz2 = dL_da2 * da2_dz2                    # dL/dz2
        
        dz2_dW2 = self.a1                            # dz2/dW2
        dL_dW2 = dz2_dW2.T @ dL_dz2                  # dL/dW2
        dL_db2 = dL_dz2.sum(axis=0)                  # dL/db2

        dz2_da1 = self.W2                            # dz2/da1
        dL_da1 = dL_dz2 @ dz2_da1.T                  # dL/da1
        da1_dz1 = self.sigmoid_derivative(self.z1)   # da1/dz1
        dL_dz1 = dL_da1 * da1_dz1                    # dL/dz1
        
        dz1_dW1 = X                                  # dz1/dW1
        dL_dW1 = dz1_dW1.T @ dL_dz1                  # dL/dW1
        dL_db1 = dL_dz1.sum(axis=0)                  # dL/db1

        # Update parameters using gradient descent
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2

    def train(self, X, y, num_epochs=10000):
        """ Training loop """
        for epoch in range(num_epochs):
            # Forward pass
            output = self.forward(X)
            # Compute loss
            loss = ((output - y) ** 2).mean()
            # Backward pass
            self.backward(X, y)
            # Print loss every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        """ Prediction (binary output) """
        with torch.no_grad():
            output = self.forward(X)
            return (output >= 0.5).float()
