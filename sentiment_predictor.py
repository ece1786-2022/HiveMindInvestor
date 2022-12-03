# -*- coding: utf-8 -*-
"""sentimentpredictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tPphoepyXnumAwnCzDUga2f8iUwC4e5P
"""

# !pip install transformers datasets
# !pip install evaluate

import numpy as np
import pandas as pd
from datetime import datetime
import torch
import re
import string
import nltk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
nltk.download('vader_lexicon')
import random
import datasets
from torch.utils.data import DataLoader

class SentimentPredictor():
  def __init__(self, input_data, model, tokenizer):
      if isinstance(input_data, pd.DataFrame):
            self.raw = input_data
      elif isinstance(input_data, str):
            self.raw = self.read_csv(input_data)
            
      self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
      
      self.model_name = model
      self.tokenizer_name = tokenizer
          # Load pretrained model
      self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
      self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
      self.model.to(self.device)
  @staticmethod
  def read_csv(filename):
    try: 
        dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
        df = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)
    except:
        dateparse = lambda dates: pd.to_datetime(dates, format='%m/%d/%Y')
        df = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)
    return df
  
  def data_process(self):
    test_r_data = self.raw.copy()
    # print(test_r_data)
    test_r_data['Post_Concat'] = self.raw['Title'] + self.raw['Post Text'].fillna('')
    # X_list = test_r_data.loc[test_r_data['Label'] == -1].index.tolist()
    # print(X_list)
    # test_r_data = test_r_data.drop(X_list)
    # print(test_r_data)
    df = test_r_data[['Post_Concat', 'Date', 'Score']].copy()
    # df = df.sort_values(by='Date',ascending=False)
    mask = (df['Date'] > self.start_date) & (df['Date'] <= self.end_date)
    
    df_date = df.loc[mask]
    # print(df_date)
    test_data = df_date[['Post_Concat']].copy()
    test_data = test_data.dropna()
    # Convert categorical labels to numerical
    # print(test_data)
    test_data.Post_Concat = test_data.Post_Concat.str.lower()

    #Remove handlers
    test_data.Post_Concat = test_data.Post_Concat.apply(lambda x:re.sub('@[^\s]+','',x))

    # Remove URLS
    test_data.Post_Concat = test_data.Post_Concat.apply(lambda x:re.sub(r"http\S+", "", x))

    # Remove all the special characters
    test_data.Post_Concat = test_data.Post_Concat.apply(lambda x:' '.join(re.findall(r'\w+', x)))

    #remove all single characters
    test_data.Post_Concat = test_data.Post_Concat.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    test_data.Post_Concat = test_data.Post_Concat.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))

    test_dataset = datasets.Dataset.from_dict(test_data)
    test_dataset_dict = datasets.DatasetDict({"test":test_dataset})
    # print(test_dataset_dict)
    return test_dataset_dict

  def predict(self,start_date, end_date):
    self.start_date = start_date
    self.end_date = end_date
    # print(model)
    tokenizer=self.tokenizer
    model=self.model
    # Tokenize function
    def tokenize_function_test(examples):
      return tokenizer(examples["Post_Concat"], padding=True,truncation=True)
    test_dataset_dict = self.data_process()
    # print(test_dataset_dict)
    # Apply Tokenize function on test set
    tokenized_datasets_test = test_dataset_dict.map(tokenize_function_test,batched=True)
    tokenized_datasets_test = tokenized_datasets_test.remove_columns(["Post_Concat"])
    # tokenized_datasets_test = tokenized_datasets_test.rename_column("Label", "labels")
    tokenized_datasets_test.set_format("torch")
    # print(tokenized_datasets_test)
    # Dataloader
    test_dataloader = DataLoader(tokenized_datasets_test['test'], batch_size=8)
    softmax = torch.nn.Softmax(dim=1)
    ret = []
    for batch in test_dataloader:
      batch = {k: v.to(self.device) for k, v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)
      logits = outputs.logits
      # print(logits)
      prediction_prob = softmax(logits)[:,1]
      ret.append(prediction_prob.detach().cpu())
    result = torch.concat(ret,dim=-1).flatten().mean()
    return result.item()

if __name__ == "__main__":
  model_name = './model/model_juliensimon'
  input_data = './data_labeled/Meta_posts_clean.csv'
  tokenizer = "juliensimon/reviews-sentiment-analysis"
  start_date = '2022-11-18 00:00'
  end_date = '2022-11-24 23:59'
  sp = SentimentPredictor(input_data, model_name, tokenizer)
  a = sp.predict(start_date, end_date)
  stop_here=1