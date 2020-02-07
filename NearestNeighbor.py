# Assignment Information
print("Data 51100- Spring 2020")
print("Alec Peterson")
print("Programming Assignment #3")

# Import numpy for use
import numpy as np

# Import data for training and testing
data_training = "iris-training-data.csv"
data_testing = "iris-testing-data.csv"

# 2D array of floats for storing training example attribute values
training_attributes = np.loadtxt(data_training, delimiter = ',', usecols = (0,1,2,3))
# 2D array of floats for storing testing example attribute values
testing_attributes = np.loadtxt(data_testing, delimiter = ',', usecols = (0,1,2,3))
# 1D array of strings for storing training example class labels
training_labels = np.loadtxt(data_training, dtype = '<U15', delimiter = ',', usecols = (4))
# 1D array of strings for storing testing example class labels
testing_labels = np.loadtxt(data_testing, dtype = '<U15', delimiter = ',', usecols = (4))
