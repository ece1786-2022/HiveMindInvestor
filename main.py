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

def mm_fit_and_eval_test(SOURCE_FILE):
    start_date = '2022-09-08 00:00:00'
    end_date = '2022-11-24 23:59:59'
    offset_days=7
    ts = TimeSeriesPredictor(SOURCE_FILE)
    def fn(args):
        start_date, end_date = args
        ts.split_data(start_date)
        actual_trend = ts.actual_trend(start_date, end_date)

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (start_date - timedelta(offset_days)).strftime("%Y-%m-%d")
        end_date = (end_date - timedelta(offset_days)).strftime("%Y-%m-%d")
        prediction = ts.actual_trend(start_date,end_date)
        score=actual_trend/prediction
        return score
    
    ret = sliding_window(start_date, end_date, 7, 7, fn)
    ret = torch.tensor(ret)
    score = ret[ret>0].shape[0]/ret.shape[0]
    windows = get_sliding_window(start_date, end_date, 7, 7)
    return {SOURCE_FILE:{'overall_score':score, 'predictions': ret, 'time_windows':windows}}

def mt_fit_and_eval_test(SOURCE_FILE):
    start_date = '2022-09-08 00:00:00'
    end_date = '2022-11-24 23:59:59'
    
    ts = TimeSeriesPredictor(SOURCE_FILE)
    def fn(args):
        rd=np.random.rand(1)
        ts.split_data(start_date)
        actual_trend = ts.actual_trend(start_date, end_date)
        if actual_trend >= 0:
            score = 1 if rd>=0.5 else 0
        else:
            score = 1 if rd<0.5 else 0
            
        return score
    
    rets=[]
    for i in range(100):
        ret = sliding_window(start_date, end_date, 7, 7, fn)
        ret = torch.tensor(ret)
        rets.append(ret)
    rets =torch.stack(rets)
    score = (rets.sum(-1)/rets.shape[-1]).mean().item()
    windows = get_sliding_window(start_date, end_date, 7, 7)
    return {SOURCE_FILE:{'overall_score':score, 'predictions': rets.sum(0)/rets.shape[0], 'time_windows':windows}}

def ts_fit_and_eval_test(SOURCE_FILE):
    start_date = '2022-09-08 00:00:00'
    end_date = '2022-11-24 23:59:59'
    
    ts = TimeSeriesPredictor(SOURCE_FILE)
    def fn(args):
        ts.fit(start_date)
        return ts.eval(args[0], args[1])
    
    ret = sliding_window(start_date, end_date, 7, 7, fn)
    ret = torch.tensor(ret)
    score = ret[ret>0].shape[0]/ret.shape[0]
    windows = get_sliding_window(start_date, end_date, 7, 7)
    return {SOURCE_FILE:{'overall_score':score, 'predictions': ret, 'time_windows':windows}}

def fn_st_fit_and_eval(start_date, end_date, offset_days, st, ts):
    ts.split_data(start_date)
    actual_trend = ts.actual_trend(start_date, end_date)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (start_date - timedelta(offset_days)).strftime("%Y-%m-%d")
    end_date = (end_date - timedelta(offset_days)).strftime("%Y-%m-%d")
    sentiment = st.predict(start_date,end_date)

    sigmoid = lambda x: 1/(1 + np.exp(-x))
    actual_trend = sigmoid(actual_trend)

    #We use 0.5 as the threshold for postive/negative predictions
    score = (sentiment-0.5)/(actual_trend-0.5)
    if pd.isna(score):
        stophere=1
    return score

def st_fit_and_eval_test(SOURCE_FILE, sentiment_file):
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    model_name = './model/model_juliensimon'
    start_date = '2022-09-08 00:00:00'
    end_date = '2022-11-24 23:59:59'
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

def fn_combined(start_date, end_date, offset_days, st, ts, alpha, beta):
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    
    ts.fit(train_test_split=start_date)
    predicted_trend = ts.predict(start_date,end_date)
    actual_trend = ts.actual_trend(start_date, end_date)
    actual_trend =sigmoid(actual_trend)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (start_date - timedelta(offset_days)).strftime("%Y-%m-%d")
    end_date = (end_date - timedelta(offset_days)).strftime("%Y-%m-%d")
    sentiment = st.predict(start_date,end_date)

    cubic_inverse_sigmoid=lambda x: -np.log2((1-x)/x)**3
    combined_prediction = sigmoid(alpha*cubic_inverse_sigmoid(predicted_trend) + beta*cubic_inverse_sigmoid(sentiment))
    
    score = (combined_prediction-0.5)/(actual_trend-0.5)
    return score

def combined_fit_and_eval_test(SOURCE_FILE, sentiment_file, alpha, beta):
    
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    model_name = './model/model_juliensimon'
    start_date = '2022-09-08 00:00:00'
    end_date = '2022-11-24 23:59:59'
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
        ts,
        alpha,
        beta
        )
        for start_date, end_date in windows
    ]
    
    # with Pool(8) as p:
    #     ret = p.starmap(fn_combined, inputs)
    ret=[]
    for i in inputs:
        ret.append(fn_combined(*i))
        
    ret = torch.tensor(ret)
    ret = ret[~ret.isnan()]
    score = ret[ret>0].shape[0]/ret.shape[0]
    print(score)
    return {SOURCE_FILE:{'overall_score':score, 'predictions': ret, 'time_windows':windows}}

def run_mm_fit_and_eval_tests():
    SOURCE_FILES=[
            'stocks_data/AMZN.csv',
            'stocks_data/AAPL.csv',
            'stocks_data/GOOG.csv',
            'stocks_data/META.csv',
            'stocks_data/NFLX.csv',
            'stocks_data/TSLA.csv'
                  ]

    with Pool(len(SOURCE_FILES)) as p:
        out = p.map(mm_fit_and_eval_test, SOURCE_FILES)
        
    with open('mm_fit_and_eval_test', 'wb') as f:
        pickle.dump(out, f)
        
def run_mt_fit_and_eval_tests():
    SOURCE_FILES=[
            'stocks_data/AMZN.csv',
            'stocks_data/AAPL.csv',
            'stocks_data/GOOG.csv',
            'stocks_data/META.csv',
            'stocks_data/NFLX.csv',
            'stocks_data/TSLA.csv'
                  ]

    with Pool(len(SOURCE_FILES)) as p:
        out = p.map(mt_fit_and_eval_test, SOURCE_FILES)
        
    with open('mt_fit_and_eval_test', 'wb') as f:
        pickle.dump(out, f)
        
def run_ts_fit_and_eval_tests():
    SOURCE_FILES=[
            'stocks_data/AMZN.csv',
            'stocks_data/AAPL.csv',
            'stocks_data/GOOG.csv',
            'stocks_data/META.csv',
            'stocks_data/NFLX.csv',
            'stocks_data/TSLA.csv'
                  ]

    with Pool(len(SOURCE_FILES)) as p:
        out = p.map(ts_fit_and_eval_test, SOURCE_FILES)
        
    with open('ts_fit_and_eval_test', 'wb') as f:
        pickle.dump(out, f)
    
def run_st_fit_and_eval_tests():
    model_name = './model/model_juliensimon'
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    ## Run sentiment tests
    out = []
    
    input_data = './reddit_data/Amazon_posts_clean.csv'
    SOURCE_FILE='stocks_data/AMZN.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Apple_posts_clean.csv'
    SOURCE_FILE='stocks_data/AAPL.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Google_posts_clean.csv'
    SOURCE_FILE='stocks_data/GOOG.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Meta_posts_clean.csv'
    SOURCE_FILE='stocks_data/META.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Netflix_posts_clean.csv'
    SOURCE_FILE='stocks_data/NFLX.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Tesla_posts_clean.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'
    res = st_fit_and_eval_test(SOURCE_FILE, input_data)
    print(res)
    out.append(res)

    with open('st_fit_and_eval_test_out', 'wb') as f:
        pickle.dump(out, f)

def run_combined_tests():
    model_name = './model/model_juliensimon'
    tokenizer = "juliensimon/reviews-sentiment-analysis"
    ## Run sentiment tests
    out = []
    
    input_data = './reddit_data/Amazon_posts_clean.csv'
    SOURCE_FILE='stocks_data/AMZN.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data, 1,10)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Apple_posts_clean.csv'
    SOURCE_FILE='stocks_data/AAPL.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data,1,10)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Google_posts_clean.csv'
    SOURCE_FILE='stocks_data/GOOG.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data,3,1)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Meta_posts_clean.csv'
    SOURCE_FILE='stocks_data/META.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data,4,1)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Netflix_posts_clean.csv'
    SOURCE_FILE='stocks_data/NFLX.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data,10,1)
    print(res)
    out.append(res)
    
    input_data = './reddit_data/Tesla_posts_clean.csv'
    SOURCE_FILE='stocks_data/TSLA.csv'
    res = combined_fit_and_eval_test(SOURCE_FILE, input_data,1,1)
    print(res)
    out.append(res)

    with open('combined_test_out', 'wb') as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    
    # run_ts_fit_and_eval_tests()
    # run_st_fit_and_eval_tests()
    # run_combined_tests()
    run_mt_fit_and_eval_tests()
    # run_mm_fit_and_eval_tests()
    
    # with open('ts_fit_and_eval_test.txt', 'r') as f:
    #     ts_out = eval(f.readline())
    # with open('ts_fit_and_eval_test', 'rb') as f:
    #     ts_out = pickle.load(f)
    # with open('st_fit_and_eval_test_out', 'rb') as f:
    #     st_out = pickle.load(f)
    # with open('combined_test_out', 'rb') as f:
    #     comb_out = pickle.load(f)
        
        
    stophere=1
