#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
unit test module for the package
"""


# In[62]:


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import norm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import unittest
from ANN import NeuralNetwork



nn = NeuralNetwork(input_size=5, hidden_size=10, output_size=1)

# Test sigmoid function
assert nn.sigmoid(0) == 0.5
assert np.isclose(nn.sigmoid(2), 0.88079707797)
assert np.isclose(nn.sigmoid(3), 0.95257412682)

# Test relu function
assert nn.relu(0) == 0
assert nn.relu(-3) == 10 # intentionally wrong to check if unit testing is working properly or not
assert nn.relu(3) == 3

# Test sigmoid_derivative function
assert nn.sigmoid_derivative(0) == 0.25
assert nn.sigmoid_derivative(2) == 0.1049935854
assert nn.sigmoid_derivative(3) == 0.04517665973

# Test relu derivative function
assert nn.relu_derivative(2) == 1
assert nn.relu_derivative(-2) == 0
assert nn.relu_derivative(0) == 0

