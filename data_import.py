
"""
class to import dataset from API and include target variable and do priliminary data splitting
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import alpaca_trade_api as tradeapi

#from config import *
import os
import time
from sklearn.model_selection import train_test_split


class alpaca_data():
    """
    para API_KEY: represents the input of API key of Alpaca database, default is 'PKL6FLMQP9AR37P9DG3M'
    para SECRET KEY: represents the input of hte user specific secret key, default is 'P2eGMaIyezfGgSoPjD2pbafdi0wnMwncFJfdjvsy'
    para BASE_URL: represents the input of url for the api, default is 'https://paper-api.alpaca.markets'
    para start_date:starting date from which data has to be downloaded, default is '2022-01-01'
    para end_date:late date upto which data has to be downloaded, default is '2022-12-31'
    para symbol:unique identification of company, default is 'AAPL' - apple
    """    
    def __init__(self, API_KEY = 'PKL6FLMQP9AR37P9DG3M',
                 SECRET_KEY = 'P2eGMaIyezfGgSoPjD2pbafdi0wnMwncFJfdjvsy',
                 BASE_URL = 'https://paper-api.alpaca.markets', 
                 start_date = '2022-01-01', 
                 end_date = '2022-12-31',
                 symbol = 'AAPL'):
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.BASE_URL = BASE_URL
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        
        self.data = self.data_import()
        self.target_data = self.target_variable()
        self.X, self.y = self.data_gen()
    
    def data_import(self):
        """
        method to download the dataset from the tradeapi.REST method from alpaca package
        uses bars - get_bars method of alpaca trade package to download data
        returns: dataset
        """
        api = tradeapi.REST(self.API_KEY, self.SECRET_KEY, base_url=self.BASE_URL, api_version='v2')
        timeframe = tradeapi.rest.TimeFrame.Day  # 1 day

        # Retrieve the historical data for the symbol and timefram
        bars = api.get_bars(self.symbol, timeframe, self.start_date, self.end_date, adjustment='raw')

        # Convert the response to a Pandas DataFrame
        data = pd.DataFrame([{
            'timestamp': bar.t,
            'open': bar.o,
            'high': bar.h,
            'low': bar.l,
            'close': bar.c,
            'volume': bar.v
        } for bar in bars])

        # Set the timestamp as the index
        data.set_index('timestamp', inplace=True)

        return data

    def target_variable(self):
        """
        method to calculate target variable by considering change of close rates
        returns: X dataset with target variable y with class 1 and 0
        """
        # target_variable = x
        x = self.data.copy()

        # calculate percent change - today's close vs tomorrow's close
        x['percent_change'] = x['close'].pct_change()

        # make target variable - 1 where percent change was positive
        x['target'] = np.where(x['percent_change']>0,1,0)

        # drop first row since it'll have nan in percent change column
        x  = x.dropna()
        return x

    def data_gen(self):
        """
        returns: X - predictor variables, y - target variable
        """
        x = self.target_data.copy()
        # self.data = x
        X = x.drop(['target', 'percent_change'], axis = 1)
        y = x['target']
        #X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.1)
        # return X_train, X_test, y_train, y_test
        return X, y

