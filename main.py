#! main.py

#Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import mnist digit recognition dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Function to visualize digits
def plot_digits(data, title, labels):
    fig, axes = plt.subplots(1, 11, figsize=(20, 2))
    fig.suptitle(title, fontsize=16)

    # Plotting the images
    for i in range(10):
        ax = axes[i]
        ax.imshow(np.reshape(data[i], (8, 8)), cmap='gray')
        ax.axis('off')

    # Plotting the distribution of digits
    ax = axes[10]
    ax.hist(labels, bins=np.arange(11) - 0.5, rwidth=0.8)
    ax.set_title('Digit Distribution')
    ax.set_xticks(range(10))
    plt.tight_layout()

    plt.show()

# Import mnist digit recognition dataset
digits = load_digits()
df = pd.DataFrame(digits.data)
print(df.head())

##VAnilla
data = np.array(df)
m, n = data.shape
print(m,n)
np.random.shuffle(data) 


data_dev = data[0:1000]
y_dev = data_dev[:, 0]
X_dev = data_dev[:, 1:].T 

data_train = data[1000:m]
y_train = data_train[:, 0]
X_train = data_train[:, 1:].T  



## NN
def init_params():
    W1 = np.random.rand(10, 63) - 0.5  
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def sigmoid(Z):
    return 1/(1+np.exp(-Z))
    
def relu(Z):
    return np.maximum(0, Z)
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def onehot(Y):
    onehot_Y = np.zeros((Y.size, int(Y.max()) + 1))
    onehot_Y[np.arange(Y.size), Y.astype(int)] = 1
    onehot_Y = onehot_Y.T
    return onehot_Y

def deriv_relu(Z):
    return Z > 0
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    onehot_Y = onehot(Y)
    dZ2 = A2 - onehot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2
def accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
def predictions(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # Adjust alpha after every 10% of the total iterations
        if i % (iterations // 10) == 0 and i != 0:
            alpha /= 10
            print(f"Reduced learning rate to: {alpha}")

        if i % 10 == 0:
            print("Iteration:", i)
            print("Accuracy:", accuracy(predictions(A2), Y))
    return W1, b1, W2, b2

# Reshape the labels to be column vectors
y_train = y_train.reshape(1, y_train.size)
y_dev = y_dev.reshape(1, y_dev.size)

W1, b1, W2, b2 = gradient_descent(X_train, y_train, 10000, 0.1)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.228, random_state=69)


plot_digits(digits.data, "Original Data", digits.target)

# Visualize training data
plot_digits(X_train, "Training Data", y_train)

# Visualize test data
plot_digits(X_test, "Test Data", y_test)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
### TORCH


# Preprocess and reshape data
X_train = X_train.reshape(-1, 1, 8, 8).astype(np.float32) / 16.0
X_test = X_test.reshape(-1, 1, 8, 8).astype(np.float32) / 16.0

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the Keras CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training loop
for epoch in range(1000):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update the learning rate

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy:', 100 * correct / total)
