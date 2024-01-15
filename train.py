##Required packages
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
#if_using_ddpg = False
if_using_ppo = False
if_using_td3 = False
if_using_sac = False

model_a2c = agent.get_model('a2c')
#model_ddpg = agent.get_model('ddbg')
model_ppo = agent.get_model('ppo')
model_td3 = agent.get_model('td3')
model_sac = agent.get_model('sac')

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_a2c.set_logger(new_logger_a2c)
'''
if if_using_ddpg:
  # set up logger
  tmp_path = RESULTS_DIR + '/ddpg'
  new_logger_ddpg = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
  # set new logger 
  model_ddbg.set_logger(new_logger_ddpg)
'''
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
'''
trained_ddpg = agent.train_model(model=model_ddpg,
                                tb_log_name= 'ddpg',
                                total_timesteps=50000) if if_using_ddpg else None
'''
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
#trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None
