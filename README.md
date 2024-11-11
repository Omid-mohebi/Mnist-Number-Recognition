# MNIST Logistic Regression Training

This repository contains a simple logistic regression implementation for training on the MNIST dataset. The script reads data from a CSV file, preprocesses it, defines a neuron, and trains it to recognize handwritten digits.
 
## Dependencies

Make sure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib
```

## Usage

1. Place the MNIST training CSV file in the `MNIST_CSV` directory and name it `mnist_train.csv`.
2. Run the script to train the neural network.

## Code Overview

### Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Reading and Preprocessing Data

```python
raw_data = pd.read_csv('../MNIST_CSV/mnist_train.csv')
data = raw_data.to_numpy()
np.random.shuffle(data)
Y = data[1:, 0]
X = data[1:, 1:] / 255
N, D = X.shape
K = len(set(Y))
```

### One-Hot Encoding Function

```python
def one_hard(local_data, n, k):
    oh_Y = np.zeros((n, k))
    for i in range(n):
        oh_Y[i, local_data[i]] = 1
    return oh_Y
```

### Initialize Parameters

```python
Y = one_hard(Y, N, K)
W = np.random.randn(D, K)
B = np.random.randn(K)
```

### Define Neural Network Functions

```python
def softmax(local_data):
    expa = np.exp(local_data)
    return expa / expa.sum(axis=1, keepdims=True)

def forward(x, w, b):
    return softmax(x.dot(w) + b)

def loss_function(py, t):
    epsilon = 1e-12
    py = np.clip(py, epsilon, 1 - epsilon)
    return -(t * np.log(py)).sum()

def predict(py):
    return np.argmax(py, axis=1)
```

### Training Loop

```python
epochs = 400
l_rate = 0.1
loss_rate = []

for i in range(epochs):
    py = forward(X, W, B)
    loss = loss_function(py, Y)
    loss_rate.append(loss)
    W -= l_rate * X.T.dot(py - Y) / N
    B -= l_rate * (py - Y).sum(axis=0) / N
```

### Plotting Loss

```python
plt.plot(loss_rate)
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```

### Evaluating Accuracy

```python
p = predict(py)
y = predict(Y)
res = np.sum(p == y) / N
print(f"Accuracy: {res * 100:.2f}%")
```

## Results

With the current hyperparameters, the model achieves approximately 70% accuracy. Increasing the number of epochs may improve accuracy.

![image](https://github.com/user-attachments/assets/5de7a6b1-6173-48b3-9d07-35c484bbbf95)


### UPDATE: Mini Batch Implementation (src_mini_batch.ipynb)

This model was trained using mini batches with a batch size of 1024. While this approach yielded the 91% accuracy, it took more time to complete (approximately 3 minutes overall, 2 minute more than logistic regression).
The only part which is different is:

```python
batch_size = 1024
batch = math.ceil(N/batch_size)

py = np.zeros(Y.shape)
for i in range(epochs):
    for j in range(batch):
        x = X[j*batch_size:(j+1)*batch_size,:]
        y = Y[j*batch_size:(j+1)*batch_size,:]
        py[j*batch_size:(j+1)*batch_size] = forward(x, W, B)
        W -= l_rate * x.T.dot(py[j*batch_size:(j+1)*batch_size] - y) / len(x)
        B -= l_rate * (py[j*batch_size:(j+1)*batch_size] - y).sum(axis=0) / len(x)
    loss = loss_function(py[j*batch_size:(j+1)*batch_size], y)
    loss_rate.append(loss)
```

![image](https://github.com/user-attachments/assets/1875dd7a-6cc3-4bff-9d1c-12052491739d)


