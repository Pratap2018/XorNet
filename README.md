## XorGate using Neural Network from scratch (Very old Code Written before i know about github arround 2017)

## Export environment

```bash
pip list --format=freeze > requirements.txt

```

```bash
conda list -e > requirements.conda.txt

```

## Import environment

```bash
pip install -r requirements.txt

```

```bash
conda create --name <env> --file requirements.txt

```




### Backward pass (manual gradients computation) with detailed explanations  (by GPT)

```python
        # 1. Output Layer
        # dL/da2: Derivative of the loss function (Mean Squared Error) with respect to the output activations (a2).
        dL_da2 = 2 * (self.a2 - y) / y.size(0)  

        # da2/dz2: Derivative of the sigmoid activation function at z2.
        da2_dz2 = self.sigmoid_derivative(self.z2)

        # dL/dz2: Chain rule: dL/da2 * da2/dz2
        dL_dz2 = dL_da2 * da2_dz2

        # dz2/dW2: Derivative of z2 with respect to the weights W2.  It's simply the activations of the previous layer (a1).
        dz2_dW2 = self.a1

        # dL/dW2: Chain rule: (dz2/dW2).T @ dL/dz2 (Matrix multiplication for multiple samples)
        dL_dW2 = dz2_dW2.T @ dL_dz2

        # dL/db2: Sum of dL/dz2 across all samples (derivatives of bias are summed).
        dL_db2 = dL_dz2.sum(axis=0)

        # 2. Hidden Layer
        # dz2/da1: Derivative of z2 with respect to the activations a1. It's just the weights W2.
        dz2_da1 = self.W2

        # dL/da1: Chain rule: dL/dz2 @ (dz2/da1).T (Matrix multiplication for multiple samples)
        dL_da1 = dL_dz2 @ dz2_da1.T  

        # da1/dz1: Derivative of the sigmoid activation function at z1.
        da1_dz1 = self.sigmoid_derivative(self.z1)

        # dL/dz1: Chain rule: dL/da1 * da1/dz1
        dL_dz1 = dL_da1 * da1_dz1

        # dz1/dW1: Derivative of z1 with respect to the weights W1. It's the input data (X).
        dz1_dW1 = X

        # dL/dW1: Chain rule: (dz1/dW1).T @ dL/dz1  (Matrix multiplication for multiple samples)
        dL_dW1 = dz1_dW1.T @ dL_dz1

        # dL/db1: Sum of dL/dz1 across all samples.
        dL_db1 = dL_dz1.sum(axis=0)

        # Update parameters (Gradient Descent)
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2

```