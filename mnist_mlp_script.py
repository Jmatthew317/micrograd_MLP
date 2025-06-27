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

