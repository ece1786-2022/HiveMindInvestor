#%%
from datetime import datetime, timedelta
import pandas as pd
from time_series_predictor import TimeSeriesPredictor
from sentiment_predictor import SentimentPredictor
import torch
import numpy as np
import pickle
from multiprocessing import Pool

def sliding_window(start_date, end_date,window_size, stride, fn):
    start_date = datetime.strptime(start_date, "%Y-%m-%d %X")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %X")
    ret=[]
    for i in range(0, (end_date-start_date).days, stride):
        s = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        e = (start_date + timedelta(days=i) + timedelta(days=window_size))
        e = (end_date).strftime("%Y-%m-%d") if e > end_date else e.strftime("%Y-%m-%d")
        score = fn((s,e))
        if not pd.isna(score):
            ret.append(score)     
    return ret

def get_sliding_window(start_date, end_date,window_size, stride):
    start_date = datetime.strptime(start_date, "%Y-%m-%d %X")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %X")
    inputs = []
    for i in range(0, (end_date-start_date).days, stride):
        s = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        e = (start_date + timedelta(days=i) + timedelta(days=window_size))
        e = (end_date).strftime("%Y-%m-%d") if e > end_date else e.strftime("%Y-%m-%d")
        inputs.append((s,e))
    return inputs

def ts_fit_and_eval_test(SOURCE_FILE):
    start_date = '2022-06-15 00:00:00'
    end_date = '2022-09-30 23:59:59'
    
    ts = TimeSeriesPredictor(SOURCE_FILE)
    def fn(args):
        ts.fit(start_date)
        return ts.eval(args[0], args[1])
    
    ret = sliding_window(start_date, end_date, 7, 1, fn)
    ret = torch.tensor(ret)
    score = ret[ret>0.5].shape[0]/ret.shape[0]
    print(score)
    return {SOURCE_FILE:score}

def fn_st_fit_and_eval(start_date, end_date, offset_days, st, ts):
    sentiment = st.predict(start_date,end_date)

    ts.split_data(start_date)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (start_date + timedelta(offset_days)).strftime("%Y-%m-%d")
    end_date = (end_date + timedelta(offset_days)).strftime("%Y-%m-%d")

    stock_trend = ts.acutal_trend(start_date, end_date)

    sigmoid = lambda x: 1/(1 + np.exp(-x))
    stock_trend = sigmoid(stock_trend)

    #We use 0.5 as the threshold for postive/negative predictions
    score = (sentiment-0.5)/(stock_trend-0.5)
    if pd.isna(score):
        stophere=1
    return score

def st_fit_and_eval_test(args):
    SOURCE_FILE, sentiment_file = args
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    model_name = './model/model_juliensimon'
    start_date = '2022-09-01 00:00:00'
    end_date = '2022-11-17 23:59:59'
    offset_days = 7
    st = SentimentPredictor(sentiment_file, model_name, tokenizer)
    ts = TimeSeriesPredictor(SOURCE_FILE)
    
    # ret = sliding_window(start_date, end_date, 7, offset_days, fn)
    windows = get_sliding_window(start_date, end_date, 7, offset_days)
    inputs = [
        (
        start_date,
        end_date,
        offset_days,
        st,
        ts
        )
        for start_date, end_date in windows
    ]
    
    with Pool(8) as p:
        ret = p.starmap(fn_st_fit_and_eval, inputs)
    ret = torch.tensor(ret)
    ret = ret[~ret.isnan()]
    score = ret[ret>0].shape[0]/ret.shape[0]
    print(score)
    return {SOURCE_FILE:{'overall_score':score, 'predictions': ret, 'time_windows':windows}}
    
if __name__ == "__main__":
    SOURCE_FILES=[
            'stocks_data/AAPL.csv',
            'stocks_data/AMZN.csv',
            'stocks_data/GOOG.csv',
            'stocks_data/META.csv',
            'stocks_data/NFLX.csv',
            'stocks_data/TSLA.csv'
                  ]
    # SOURCE_FILE='stocks_data/AAPL.csv'
    # SOURCE_FILE='stocks_data/AMZN.csv'
    # SOURCE_FILE='stocks_data/GOOG.csv'
    # SOURCE_FILE='stocks_data/META.csv'
    # SOURCE_FILE='stocks_data/NFLX.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'
    
    ##Run timeseries tests
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
    
    # out={}
    # for f in SOURCE_FILES:
    #     out.update(ts_fit_and_eval_test(f))

    # with Pool(len(SOURCE_FILES)) as p:
    #     out = p.map(ts_fit_and_eval_test, SOURCE_FILES)
        
    # with open('ts_fit_and_eval_test.txt', 'w') as f:
    #     f.write(str(out))

    # 
    model_name = './model/model_juliensimon'
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    
    # SOURCE_FILE='stocks_data/AAPL.csv'
    # SOURCE_FILE='stocks_data/AMZN.csv'
    # SOURCE_FILE='stocks_data/GOOG.csv'
    # SOURCE_FILE='stocks_data/META.csv'
    # SOURCE_FILE='stocks_data/NFLX.csv'
    ## Run sentiment tests
    out = []
    
    input_data = './reddit_data/Amazon_posts_clean.csv'
    SOURCE_FILE='stocks_data/AMZN.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Apple_posts_clean.csv'
    SOURCE_FILE='stocks_data/AAPL.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Google_posts_clean.csv'
    SOURCE_FILE='stocks_data/GOOG.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Meta_posts_clean.csv'
    SOURCE_FILE='stocks_data/META.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Netflix_posts_clean.csv'
    SOURCE_FILE='stocks_data/NFLX.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Tesla_posts_clean.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'
    res = st_fit_and_eval_test((SOURCE_FILE, input_data))
    print(res)
    out.append(res)

    with open('s_fit_and_eval_test_out', 'w') as f:
        pickle.dump(out, f)
