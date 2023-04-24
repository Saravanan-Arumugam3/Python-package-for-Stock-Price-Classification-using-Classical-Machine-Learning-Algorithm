
"""
class to execute the machine learning algorithms for the radeing dataset and generate the performace metrics
"""
from ANN import NeuralNetwork
import data_import as api
from LogisticReg import Logistic_Regression

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


data = api.alpaca_data(API_KEY = 'PKL6FLMQP9AR37P9DG3M',
             SECRET_KEY = 'P2eGMaIyezfGgSoPjD2pbafdi0wnMwncFJfdjvsy',
             BASE_URL = 'https://paper-api.alpaca.markets', 
             start_date = '2022-01-01', 
             end_date = '2022-12-31',
             symbol = 'AAPL')



# model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=2, output_size=1)
# model.train(X= X_train, y=y_train.to_numpy(), num_epochs=10, learning_rate=0.01)
print("need fixing")


model = Logistic_Regression(X,y)
model.fit()
model.performance_metrics()
model.y_prediction


    
