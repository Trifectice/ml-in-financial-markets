#Required packages
#!pip install swig
#!pip install wrds
#!pip install pyportfolioopt
## Instal finrl library
#!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git


#imports

import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader 
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

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

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list= INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature= False)

processed = fe.preprocess_data(df_raw)