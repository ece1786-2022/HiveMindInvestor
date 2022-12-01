#%%
from datetime import datetime, timedelta
import pandas as pd

def sliding_window(start_date, end_date,window_size, stride, fn):
    start_date = datetime.strptime(start_date, "%Y-%m-%d %X")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %X")
    ret=[]
    for i in range(0, (end_date-start_date).days, stride):
        s = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        e = (start_date + timedelta(days=i) + timedelta(days=window_size)).strftime("%Y-%m-%d")
        score = fn((s,e))
        if not pd.isna(score):
            ret.append(score)     
    return ret

if __name__ == "__main__":
    from time_series_predictor import TimeSeriesPredictor
    import torch
    # SOURCE_FILE='stocks_data/AAPL.csv'
    # SOURCE_FILE='stocks_data/AMZN.csv'
    # SOURCE_FILE='stocks_data/GOOG.csv'
    # SOURCE_FILE='stocks_data/META.csv'
    # SOURCE_FILE='stocks_data/NFLX.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'

    # tsp = TimeSeriesPredictor(SOURCE_FILE, train_test_split=0.9)
    #===
    # tsp = TimeSeriesPredictor(SOURCE_FILE, train_test_split='2022-06-01')
    #===
    # df = TimeSeriesPredictor.parse_csv(SOURCE_FILE)
    # mask = (df['ds'] > '2021-01-01')
    # df = df.loc[mask]
    # tsp = TimeSeriesPredictor(df, train_test_split=0.8)
    
    # tsp.fit()
    # #%%
    # tsp.plot_fit()
    #%%
    
    # fn = lambda x: tsp.fit(x[0], x[1])
    
    fn = lambda x: TimeSeriesPredictor.fit_and_eval(SOURCE_FILE, x[0], x[1])
    
    start_date = '2022-06-15 00:00:00'
    end_date = '2022-09-30 23:59:59'
    ret = sliding_window(start_date, end_date, 7, 1, fn)
    ret = torch.tensor(ret)
    score = ret[ret>0.5].shape[0]/ret.shape[0]
    print(score)

# %%
