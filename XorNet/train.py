
import torch
from nn import XORNet

# XOR truth table
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Create and train the network
model = XORNet()
model.train(X, y)



# Test the network
predictions = model.predict(X)
print("\nPredictions:")
print(predictions)