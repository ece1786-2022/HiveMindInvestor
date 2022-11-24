
#%%
# Python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
import torch


class TimeSeriesPredictor():
    def __init__(self, input_data, train_test_ratio=0.9):
        if isinstance(input_data, pd.DataFrame):
            self.raw = input_data
        elif isinstance(input_data, str):
            self.raw = self.parse_csv(input_data)

        self.train_data, self.test_data = self.raw[3:int(len(self.raw)*0.90)], self.raw[int(len(self.raw)*0.90):]
    
    def fit(self):
        m = Prophet(daily_seasonality = True)
        m.fit(self.train_data)
        self.m = m
        
        m=self.m
        test_data = self.test_data
        df=self.raw
        
        length = (self.test_data['ds'].iloc[-1] - self.test_data['ds'].iloc[0]).days
        future = m.make_future_dataframe(periods=length)


        # Python
        fc = m.predict(future)
        fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        self.fc = fc
    
    def predict(self,start_date,end_date, temp=1.0):
        assert (start_date > self.test_data['ds']).sum() > 0
        assert  (end_date < self.test_data['ds']).sum() > 0
    
        fc=self.fc
        
        mask = (fc['ds'] > start_date) & (fc['ds'] <= end_date)
        fc = fc.loc[mask]
        mask = (self.test_data['ds'] > start_date) & (self.test_data['ds'] <= end_date)
        test_data = self.test_data.loc[mask]
        
        predicted_trend = (fc['yhat'].iloc[-1] - fc['yhat'].iloc[0])*temp/(fc['ds'].iloc[-1] - fc['ds'].iloc[0]).days
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        return sigmoid(predicted_trend)
            
    def eval(self,start_date, end_date, loss_fn = None):
        '''
        loss_fn takes a predicted_trend and a actual_trend as inpts
        '''
        assert (start_date > self.test_data['ds']).sum() > 0
        assert  (end_date < self.test_data['ds']).sum() > 0
    
        fc=self.fc
        
        mask = (fc['ds'] > start_date) & (fc['ds'] <= end_date)
        fc = fc.loc[mask]
        mask = (self.test_data['ds'] > start_date) & (self.test_data['ds'] <= end_date)
        test_data = self.test_data.loc[mask]
        
        predicted_trend = fc['yhat'].iloc[-1] - fc['yhat'].iloc[0]
        actual_trend = test_data['y'].iloc[-1] - test_data['y'].iloc[0]
        
        score=None
        if loss_fn is None:
            sigmoid = lambda x: 1/(1 + np.exp(-x))
            score = sigmoid(predicted_trend/actual_trend)
        else:
            score = loss_fn((predicted_trend, actual_trend))
        return 
        
        
    def plot_fit(self):
        train_data = self.train_data
        test_data = self.test_data
        df = self.raw
        fc = self.fc
        fc_series = pd.Series(fc['yhat'].to_numpy())
        lower_series = pd.Series(fc['yhat_lower'].to_numpy())
        upper_series = pd.Series(fc['yhat_upper'].to_numpy())

        plt.figure(figsize=(10,5), dpi=100)
        plt.plot(train_data['ds'], train_data['y'], label='training data')
        plt.plot(test_data['ds'], test_data['y'], color = 'blue', label='Actual Stock Price')
        plt.plot(df['ds'].iloc[3:],fc_series, color = 'orange',label='Predicted Stock Price')
        plt.fill_between(df['ds'].iloc[3:], lower_series, upper_series, 
                        color='k', alpha=.10)
        plt.title(' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(' Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

    @staticmethod
    def parse_csv(filename):
        try: 
            dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
            df = pd.read_csv(filename,sep=',', parse_dates=['Date'], date_parser=dateparse).fillna(0)
        except:
            dateparse = lambda dates: pd.to_datetime(dates, format='%m/%d/%Y')
            df = pd.read_csv(filename,sep=',', parse_dates=['Date'], date_parser=dateparse).fillna(0)
        df = df[['Date', 'Close']]
        df = df.rename(columns={"Date":"ds", "Close":"y"})
        return df

if __name__ == "__main__":
    # SOURCE_FILE='stocks_data/AAPL.csv'
    # SOURCE_FILE='stocks_data/AMZN.csv'
    # SOURCE_FILE='stocks_data/GOOG.csv'
    # SOURCE_FILE='stocks_data/META.csv'
    # SOURCE_FILE='stocks_data/NFLX.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'
    
    #%%
    tsp = TimeSeriesPredictor(SOURCE_FILE,0.9)
    tsp.fit()
    #%%
    start_date = '2022-08-01 00:00:00'
    end_date = '2022-08-08 23:59:59'
    print(tsp.predict(start_date, end_date, temp=1.0))
    print(tsp.eval(start_date, end_date))

# %%
