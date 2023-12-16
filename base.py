#Required packages
#pip install swig
#pip install wrds
#pip install pyportfolioopt
## Instal finrl library
#pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


#imports

import os
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader 
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR

import itertools

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2023-05-01'

symbols = [
    'aapl',
    'msft',
    'meta',
    'ibm',
    'hd',
    'cat',
    'amzn',
    'intc',
    't',
    'v',
    'gs'
]

df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                         end_date= TRADE_END_DATE,
                         ticker_list= symbols).fetch_data()

#Preprocess Data

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list= INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature= False)

processed = fe.preprocess_data(df_raw)

list_ticker = processed['tic'].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=['date', 'tic']).merge(processed, on=['date','tic'],how='left')
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])

processed_full = processed_full.fillna(0)

##Save the Data
#Split the data
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(len(train))
print(len(trade))


train_path = './data-sets/train_data.csv'
trade_path = './data-sets/trade_data.csv'

with open(train_path, 'w', encoding='utf-8-sig') as f:
    train.to_csv(f)

with open(trade_path, 'w', encoding='utf-8-sig') as f:
    trade.to_csv(f)

check_and_make_directories([TRAINED_MODEL_DIR]) 

#Pull data from data-sets
train = pd.read_csv('./data-sets/train_data.csv')
train = train.set_index(train.columns[0])
train.index.names = ['']

#Setup Environment
stock_dimensions = len(train.tic.unique())
state_space = 1 + 2*stock_dimensions + len(INDICATORS)*stock_dimensions
print(f'Stock Dimension: {stock_dimensions}, State Space: {state_space}')