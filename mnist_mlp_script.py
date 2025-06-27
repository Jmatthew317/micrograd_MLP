#
# Welcome to my first attempt at writing an MLP referencing another library that I wrote 
# All credit to the original micrograd MLP goes to Andrej Karpthy
# I followed his videos and wrote it alongside his instruction
#

# Import my library, numpy, csv and pickle 
from micrograd_mlp import MLP, Value
import numpy as np
import csv
import pickle

# Importing and prepping the dataset
def load_mnist_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    data = np.array(data, dtype=np.float32) # This imports all the data and converts it from a string value to float32
    x = data[:, 1:] / 255.0 # Normalize pixels to 0-1(
    y = data[:, 0].astype(int) # create the Label
    return x, y

# Splitting up the dataset into their x and y sets
X_train, y_train = load_mnist_csv("../micrograd_MLP_data/MNIST_CSV/mnist_train.csv")
X_test, y_test = load_mnist_csv("../micrograd_MLP_data/MNIST_CSV/mnist_test.csv")

# Verify that the datasets have the right shape 
# print("Training set:", X_train.shape, y_train.shape)
# print("Test set:", X_test.shape, y_test.shape)

# Initialize the model based off the dataset parameters
model = MLP(784, [128, 64, 10])

# Test the steps and show the initial output
# sample_input = [Value(v) for v in X_train[0]]
# output = model(sample_input)
# print("Model output:", [o.data for o in output])
# print("True lable:", y_train[0])

# First forward pass in training loop
for epoch in range(1):
    total_loss = 0.0

    for i in range(100):
        # Wrap up input in Value
        x = [Value(v) for v in X_train[i]]

        # Need to convert the label to one-hot
        y_true = [0.0] * 10
        y_true[y_train[i]] = 1.0

        # Forward Pass
        y_pred = model(x)

        # Compute loss: Mean Squared Error (MSE)
        loss = sum((yp - yt)**2 for yp, yt in zip(y_pred, y_true))
        total_loss += loss.data

        # Backward pass
        for p in model.parameters():
            p.grad = 0.0 # make sure to zero the gradient before backprop
        loss.backward()

        # Gradient descent step
        for p in model.parameters():
            p.data -= 0.05 * p.grad # sets learning rate to 0.05

        # print interval outputs to monitor
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.data:.4f}")

print(f"Epoch {epoch+1} complete | Average Loss: {total_loss / 100:.4f}")


