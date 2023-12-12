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
