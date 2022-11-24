from datetime import datetime, timedelta

def sliding_window(start_date, end_date,window_size, stride, fn):
    start_date = datetime.strptime(start_date, "%Y-%m-%d %X")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %X")
    ret=[]
    for i in range(0, (end_date-start_date).days, stride):
        s = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        e = (start_date + timedelta(days=i) + timedelta(days=window_size)).strftime("%Y-%m-%d")
        ret.append(fn((s,e)))
    return ret

if __name__ == "__main__":
    from time_series_predictor import TimeSeriesPredictor
    import torch
    # SOURCE_FILE='stocks_data/AAPL.csv'
    # SOURCE_FILE='stocks_data/AMZN.csv'
    # SOURCE_FILE='stocks_data/GOOG.csv'
    # SOURCE_FILE='stocks_data/META.csv'
    SOURCE_FILE='stocks_data/NFLX.csv'
    # SOURCE_FILE='stocks_data/TSLA.csv'

    tsp = TimeSeriesPredictor(SOURCE_FILE, train_test_ratio=0.9)
    tsp.fit()

    start_date = '2022-6-01 00:00:00'
    end_date = '2022-10-01 23:59:59'

    fn = lambda x: tsp.eval(x[0], x[1])
    ret = sliding_window(start_date, end_date, 7, 7, fn)
    ret = torch.tensor(ret)
    score = ret[ret>0.5].shape[0]/ret.shape[0]
    print(score)
