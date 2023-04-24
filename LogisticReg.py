
"""
Logistic Regression - Classic Machine Learning Classification Algorithm as a Class
"""

#!/usr/bin/env python
# coding: utf-8

# In[3]:
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


class Logistic_Regression:
    def __init__(self, X, y, learning_Rate = 0.01, max_Iteration = 10000, tolerance = 0.0001):
        """
        para X: dataset for training, predictor variables
        para y: target variable
        para learning_rate: alpha/learning rate of the gradient descent 
        para max_iteration: maximum number of iteratios to execute before stopping
        para tolerance: tolerance of multicolinearity
        """
        self.learning_Rate = learning_Rate
        self.max_Iteration = max_Iteration
        self.tolerance = tolerance
        self.X = X
        self.y = y
        

    def data_split(self):
        """
        method to split the dataset into train, test using sklearn train_test_split method
        returns: X_train, X_test, y_train, y_test - datasets split for testing and training
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.03, random_state=3, shuffle=True)
        return X_train, X_test, y_train, y_test
    
    def normalize_Train(self, X):
        """
        para X: dataset for training, predictor variables
        returns: X_norm, mean, standard deviation of X
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean)/std
        return X_norm, mean, std

    def normalize_Test(self, X, mean, std):
        """
        para X: dataset for testing
        para mean: average of the dataset
        para std: standard deviation of X
        returns: X_norm - normalized dataset
        """
        X_norm = (X - mean)/ std
        return X_norm
    
    def add_X0(self, X):
        """
        method to add bias term
        returns: dataset with new column filled with 0
        """
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def sigmoid(self, z):
        """
        method to calculate signmiod function
        para z: x (prerdictor variable) multiplied by w (weights)
        returns: output of sigmiod function
        """
        sig = 1 / (1+ np.exp(-z))
        return sig
    
    def cost_Function(self, X, y):
        """
        para X: dataset for training, predictor variables
        para y: target variable
        returns: cost of regression
        """
        sig = self.sigmoid(X.dot(self.w))
        loss = y * np.log(sig) + (1-y) * np.log(1-sig)
        cost = - loss.sum()
        return cost

    def gradient(self, X, y):
        """
        para X: dataset for training, predictor variables
        para y: target variable
        returns: gradient of sigmoid fufnction
        """
        sig = self.sigmoid(X.dot(self.w))
        grad = (sig-y).dot(X)
        return grad
    
    def predict(self, X):
        """
        predict method to predict the class of test and train dataset
        para X: dataset both train and test
        returns: prediction class of y
        """
        z = X.dot(self.w)
        y_prediction = self.sigmoid(z)
        return np.around(y_prediction)
    
    def predict_prob(self, X):
        """
        predict method to predict probabilty the class of test and train dataset
        para X: dataset both train and test
        returns: prediction probabilty of class of y using sigmoid 
        """
        z = X.dot(self.w)
        y_prediction = self.sigmoid(z)
        return y_prediction
   
    
    def gradient_Descent(self, X, y):
        """
        para X: training dataset X, predictor variables
        para y: target variable of dataset 
        """
        losses = []
        prev_loss = float('inf')

        for i in tqdm(range(self.max_Iteration), colour = 'black'):
            self.w = self.w - self.learning_Rate * self.gradient(X, y)
            current_loss = self.cost_Function(X, y)
            diff_loss = np.abs(prev_loss - current_loss)
            losses.append(current_loss)

            if diff_loss < self.tolerance:
                print("The model stopped learning")
                break

            prev_loss = current_loss

    def fit(self):
        """
        method to fit/rain the model - it calculates probabilities and predictions
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_split()
        
        # Normalizing training set
        self.X_train, mean, std = self.normalize_Train(self.X_train)
        #print(mean, std)
        
        # Normalizing test set
        self.X_test = self.normalize_Test(self.X_test, mean, std)

        # Adding column of 1
        self.X_train = self.add_X0(self.X_train)
        self.X_test = self.add_X0(self.X_test)

        # Initializing weights
        self.w = np.ones(self.X_train.shape[1], dtype = np.float64)

        # Gradient Descent
        print('Solving using gradient descent')
        self.gradient_Descent(self.X_train, self.y_train)

        # Normalizing the whole Data set for prediction
        self.X_normalized = self.normalize_Train(self.X)[0]
        self.X_normalized = self.add_X0(self.X_normalized)

        # Saving Probabilities
        self.y1_prob_test = self.predict_prob(self.X_test)
        self.y1_prob = self.predict_prob(self.X_normalized)

        # Saving test predictions
        self.y_prediction_test = self.predict(self.X_test)
        self.y_prediction = self.predict(self.X_normalized)

 
    def performance_metrics(self):
        """
        method to calculate the performance merics of the algorithm, uses sklearn
        returns: accurace, precision, recall, F1 score
        """
        
        y_true = self.y_test
        y_prediction = self.y_prediction_test

        accuracy = accuracy_score(y_true, y_prediction)
        precision = precision_score(y_true, y_prediction)
        recall = recall_score(y_true, y_prediction)
        f1_score_value = f1_score(y_true, y_prediction)

        print(" ")
        print("Accuracy", accuracy)
        print("Precision", precision)
        print("Recall", recall)
        print("F1 Score", f1_score_value)
        print(" ")

        return accuracy
# In[ ]:




