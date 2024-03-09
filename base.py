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
import matplotlib.pyplot as plt 
import sys
import itertools

from pypfopt.efficient_frontier import EfficientFrontier
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader 
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR


TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2023-05-01'

symbols = [
    'aapl',
    'msft',
    'META',
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

buy_cost_ls = sell_cost_ls = [0.001 for _ in range(stock_dimensions)]
num_stock_shares = [0] * stock_dimensions

env_kwargs = {
    'hmax': 100,
    'initial_amount': 100000,
    'num_stock_shares': num_stock_shares,
    'buy_cost_pct': buy_cost_ls,
    'sell_cost_pct': sell_cost_ls,
    'state_space': state_space,
    'stock_dim': stock_dimensions,
    'tech_indicator_list': INDICATORS,
    'action_space': stock_dimensions,
    'reward_scaling': 1e-4
    }


e_train_gym = StockTradingEnv(df = train, **env_kwargs)

#Enviroment for training
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

#Train Agent

agent = DRLAgent(env = env_train)

#Mutiple Fin RL training options, multiple can be used
if_using_a2c = True
if_using_ddpg = False
if_using_ppo = False
if_using_td3 = False
if_using_sac = False

model_a2c = agent.get_model('a2c')
model_ddpg = agent.get_model('ddpg')
model_ppo = agent.get_model('ppo')
model_td3 = agent.get_model('td3')
model_sac = agent.get_model('sac')

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_a2c.set_logger(new_logger_a2c)

if if_using_ddpg:
  # set up logger
  tmp_path = RESULTS_DIR + '/ddpg'
  new_logger_ddpg = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_ddpg.set_logger(new_logger_ddpg)

if if_using_ppo:
  # set up logger
  tmp_path = RESULTS_DIR + '/ppo'
  new_logger_ppo = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_ppo.set_logger(new_logger_ppo)

if if_using_td3:
  # set up logger
  tmp_path = RESULTS_DIR + '/td3'
  new_logger_td3 = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_td3.set_logger(new_logger_td3)

if if_using_sac:
  # set up logger
  tmp_path = RESULTS_DIR + '/sac'
  new_logger_sac = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_sac.set_logger(new_logger_sac)

#training different models dependant on models boolean
trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name= 'a2c',
                                total_timesteps=50000) if if_using_a2c else None

trained_ddpg = agent.train_model(model=model_ddpg,
                                tb_log_name= 'ddpg',
                                total_timesteps=50000) if if_using_ddpg else None

trained_ppo = agent.train_model(model=model_ppo,
                                tb_log_name= 'ppo',
                                total_timesteps=50000) if if_using_ppo else None

trained_td3 = agent.train_model(model=model_td3,
                                tb_log_name= 'td3',
                                total_timesteps=50000) if if_using_td3 else None

trained_sac = agent.train_model(model=model_sac,
                                tb_log_name= 'sac',
                                total_timesteps=50000) if if_using_sac else None

#Saving trained Models
trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

#Pulling data for back testing and traiding env

trained_a2c = A2C.load('trained_models/agent_a2c.zip') if if_using_a2c else None
trained_ddpg = DDPG.load('trained_models/agent_ddpg') if if_using_ddpg else None
trained_ppo = PPO.load('trained_models/agent_ppo') if if_using_ppo else None
trained_td3 = TD3.load('trained_models/agent_td3') if if_using_td3 else None
trained_sac = SAC.load('trained_models/agent_sac') if if_using_sac else None

#Out of sample performance 

stock_dimensions = len(trade.tic.unique())
state_space = 1 + 2*stock_dimensions + len(INDICATORS)*stock_dimensions
print(f"Stock Dimentsion: {stock_dimensions}, State Space: {state_space}")

buy_cost_ls = sell_cost_ls = [0.001] * stock_dimensions
num_stock_shares = [0] * stock_dimensions

env_kwargs = {
    'hmax': 100,
    'initial_amount': 100000,
    'num_stock_shares': num_stock_shares,
    'buy_cost_pct': buy_cost_ls,
    'sell_cost_pct': sell_cost_ls,
    'state_space': state_space,
    'stock_dim': stock_dimensions,
    'tech_indicator_list': INDICATORS,
    'action_space': stock_dimensions,
    'reward_scaling': 1e-4
    }

e_train_gym = StockTradingEnv(df = trade, turbulence_threshold = 70, risk_indicator_col='vix', **env_kwargs)

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment = e_train_gym) if if_using_a2c else (None, None)
 
##Mean variance optimization (MVO)

#Help us process data into a form for weight calculation
def process_df_for_mvo(df):
    df = df.sort_values(['date', 'tic'],ignore_index=True)[['date', 'tic', 'close']]
    fst = df
    fst = fst.iloc[0:stock_dimensions, :]
    tic = fst['tic'].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0

    for i in range(df.shape[0]//stock_dimensions):
        n = df
        n = n.iloc[i * stock_dimensions:(i+1) * stock_dimensions, :]
        date = n['date'][i*stock_dimensions]
        mvo.loc[date] = n['close'].tolist()

    return mvo

#Calculates weights of average return and convariance matrix
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows-1, Columns])
    for j in range(Columns):      # j: Assets
        for i in range(Rows-1):   # i: Daily Prices
            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100
   
    return StockReturn

StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)
print(StockData)
print(TradeData)
TradeData.to_numpy()

#compute asset returns

arStockPrices = np.asarray(StockData)
[Rows, Cols]=arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

#compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis = 0)
covReturns = np.cov(arReturns, rowvar = False)

#set precision for printing results 
np.set_printoptions(precision=3, suppress = True)

#display mean returns and variance-covariance matrix of returns
print("Mean return of assets in k-portfolio 1\n", meanReturns)
print("Variance-Covariance matrix of returns\n", covReturns)

# Calculate the efficient frontier to get weights
ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(10)])

mvo_weights
# Assuming 'StockData' is a pandas DataFrame with assets as column names
asset_list = StockData.columns.tolist()


#Apply weights to previous price in stock Data
LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)


Initial_Portfolio

#Testing MVO against out of sample data
Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_results = pd.DataFrame(Portfolio_Assets, columns=['Mean Var'])

MVO_results


