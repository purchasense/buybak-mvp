"""BuyBak Appointment Booking Tool Spec."""

import numpy as np
import ta  # pip install ta
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix

from typing import Optional
from typing import ClassVar
from typing import List
from pydantic import BaseModel, Field
import json

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from typing import Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.schema import Document

import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences
from utilsforecast.plotting import plot_series

from enum import IntEnum


from llama_index.llms.openai import OpenAI


# we will represent and track the state of a booking as a Pydantic model

# url="https://www.kayak.com/flights/GOX-ORD/2025-06-04/2025-06-22/1adults/children-17-17?ucs=111myh5&sort=bestflight_a&fs=stops=1",
class BuyBakLineItemAndIndicators(BaseModel):
    """Single line item to hold the OHLCV and indicator for the list of items for a symbol"""
    open:           float = Field(description="'Open' price of that day")
    high:           float = Field(description="'High' price that day")
    low:            float = Field(description="'Low' price of that day")
    close:          float = Field(description="'Low' price of that day")
    volume:         int = Field(description="'Volume' of that day")
    rsi:            Optional[float] = Field(description="'RSI' indicator calculated based on OHLC")
    macd:           Optional[float] = Field(description="'MACD' indicator, calculated based on the last 14 rolling average")
    sma:            Optional[float] = Field(description="'SMA' indicator: simple moving average, calculated based on the last 14 rolling average")
    ema:            Optional[float] = Field(description="'EMA' indicator: exponential moving average, calculated based on the last 14 rolling average")
    volatility:     Optional[float] = Field(description="'Implied Volatility' calculated based on the last 14 rolling average")

class BuyBakTimeSeries(BaseModel):
    """A struct to hold the line_items of OHLCV and  market indicators"""
    name: str = None
    line_items: List[BuyBakLineItemAndIndicators] = Field(description="Line Items shipped with the indicators as response")

class BuyBakWines(IntEnum):
    goog = 0
    AMZN = 1
    COST = 2
    LLY  = 3
    META = 4
    TSLA = 5
    NVDA = 6
    aapl = 7
    MSFT = 8

class BuyBakTimeSeriesToolSpec(BaseToolSpec):
    """BuyBak Apt Booking Tool spec."""

    spec_functions = [
        "extract_bbk_time_series_from_prompt",
        "get_mean_squared_error",
        "buybak_model_forecast"
    ]
    mse                         = []
    storage_dir                 = "./storage-buybak-time-series"

    # XGBoost Regressor Model
    model                       = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]

    # MLForecast Regressor Model
    ml_id                       = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
    ml_ds                       = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
    ml_y                        = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
    ml_series                   = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
    ml_predict                  = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
    ml_forecaster               = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]



    def __init__(self):
        print('BBK: Initializing Time Series')

        #####################
        # reverse_goog.cv
        #####################
        df_ohlcv = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
        X = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
        y = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
        lf = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
        X_train = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]
        y_train = [ BuyBakWines.goog, BuyBakWines.AMZN, BuyBakWines.COST, BuyBakWines.LLY, BuyBakWines.META, BuyBakWines.NVDA, BuyBakWines.TSLA, BuyBakWines.aapl, BuyBakWines.MSFT ]

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.goog] = pd.read_csv("./TradesCSV/reverse_goog.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.goog]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.goog]['Close']).rsi()
        df_ohlcv[BuyBakWines.goog]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.goog]['Close']).macd()
        df_ohlcv[BuyBakWines.goog]['sma'] = df_ohlcv[BuyBakWines.goog]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.goog]['ema'] = df_ohlcv[BuyBakWines.goog]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.goog]['volatility'] = df_ohlcv[BuyBakWines.goog]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.goog].dropna(inplace=True)

        #####################
        # reverse_AMZN.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.AMZN] = pd.read_csv("./TradesCSV/reverse_AMZN.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.AMZN]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.AMZN]['Close']).rsi()
        df_ohlcv[BuyBakWines.AMZN]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.AMZN]['Close']).macd()
        df_ohlcv[BuyBakWines.AMZN]['sma'] = df_ohlcv[BuyBakWines.AMZN]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.AMZN]['ema'] = df_ohlcv[BuyBakWines.AMZN]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.AMZN]['volatility'] = df_ohlcv[BuyBakWines.AMZN]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.AMZN].dropna(inplace=True)

        #####################
        # reverse_COST.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.COST] = pd.read_csv("./TradesCSV/reverse_COST.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.COST]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.COST]['Close']).rsi()
        df_ohlcv[BuyBakWines.COST]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.COST]['Close']).macd()
        df_ohlcv[BuyBakWines.COST]['sma'] = df_ohlcv[BuyBakWines.COST]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.COST]['ema'] = df_ohlcv[BuyBakWines.COST]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.COST]['volatility'] = df_ohlcv[BuyBakWines.COST]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.COST].dropna(inplace=True)

        #####################
        # reverse_LLY.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.LLY] = pd.read_csv("./TradesCSV/reverse_LLY.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.LLY]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.LLY]['Close']).rsi()
        df_ohlcv[BuyBakWines.LLY]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.LLY]['Close']).macd()
        df_ohlcv[BuyBakWines.LLY]['sma'] = df_ohlcv[BuyBakWines.LLY]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.LLY]['ema'] = df_ohlcv[BuyBakWines.LLY]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.LLY]['volatility'] = df_ohlcv[BuyBakWines.LLY]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.LLY].dropna(inplace=True)

        #####################
        # reverse_META.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.META] = pd.read_csv("./TradesCSV/reverse_META.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.META]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.META]['Close']).rsi()
        df_ohlcv[BuyBakWines.META]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.META]['Close']).macd()
        df_ohlcv[BuyBakWines.META]['sma'] = df_ohlcv[BuyBakWines.META]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.META]['ema'] = df_ohlcv[BuyBakWines.META]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.META]['volatility'] = df_ohlcv[BuyBakWines.META]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.META].dropna(inplace=True)

        #####################
        # reverse_NVDA.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.NVDA] = pd.read_csv("./TradesCSV/reverse_NVDA.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.NVDA]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.NVDA]['Close']).rsi()
        df_ohlcv[BuyBakWines.NVDA]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.NVDA]['Close']).macd()
        df_ohlcv[BuyBakWines.NVDA]['sma'] = df_ohlcv[BuyBakWines.NVDA]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.NVDA]['ema'] = df_ohlcv[BuyBakWines.NVDA]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.NVDA]['volatility'] = df_ohlcv[BuyBakWines.NVDA]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.NVDA].dropna(inplace=True)

        #####################
        # reverse_TSLA.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.TSLA] = pd.read_csv("./TradesCSV/reverse_TSLA.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.TSLA]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.TSLA]['Close']).rsi()
        df_ohlcv[BuyBakWines.TSLA]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.TSLA]['Close']).macd()
        df_ohlcv[BuyBakWines.TSLA]['sma'] = df_ohlcv[BuyBakWines.TSLA]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.TSLA]['ema'] = df_ohlcv[BuyBakWines.TSLA]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.TSLA]['volatility'] = df_ohlcv[BuyBakWines.TSLA]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.TSLA].dropna(inplace=True)

        #####################
        # reverse_aapl.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.aapl] = pd.read_csv("./TradesCSV/reverse_aapl.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.aapl]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.aapl]['Close']).rsi()
        df_ohlcv[BuyBakWines.aapl]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.aapl]['Close']).macd()
        df_ohlcv[BuyBakWines.aapl]['sma'] = df_ohlcv[BuyBakWines.aapl]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.aapl]['ema'] = df_ohlcv[BuyBakWines.aapl]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.aapl]['volatility'] = df_ohlcv[BuyBakWines.aapl]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.aapl].dropna(inplace=True)

        #####################
        # reverse_MSFT.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.MSFT] = pd.read_csv("./TradesCSV/reverse_MSFT.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.MSFT]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.MSFT]['Close']).rsi()
        df_ohlcv[BuyBakWines.MSFT]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.MSFT]['Close']).macd()
        df_ohlcv[BuyBakWines.MSFT]['sma'] = df_ohlcv[BuyBakWines.MSFT]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.MSFT]['ema'] = df_ohlcv[BuyBakWines.MSFT]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.MSFT]['volatility'] = df_ohlcv[BuyBakWines.MSFT]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.MSFT].dropna(inplace=True)

        ##############################
        # prepare X y
        ##############################

        #######
        # goog
        #######
        lf[BuyBakWines.goog] = df_ohlcv[BuyBakWines.goog][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.goog] = np.array(lf[BuyBakWines.goog]).tolist()
        y[BuyBakWines.goog] = np.array(df_ohlcv[BuyBakWines.goog]['ema']).tolist()

        self.ml_id[BuyBakWines.goog] = np.repeat(['id_00', 'id_01'], 105)
        self.ml_ds[BuyBakWines.goog] = pd.date_range('2000-01-01', periods=210, freq='D')
        self.ml_y[BuyBakWines.goog] = df_ohlcv[BuyBakWines.goog]['ema']
        print(f'[goog] len {len(y[BuyBakWines.goog])}, {len(self.ml_id[BuyBakWines.goog])} , {len(self.ml_ds[BuyBakWines.goog])} , {len(self.ml_y[BuyBakWines.goog])} ')
        self.ml_series[BuyBakWines.goog] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.goog],
            'ds': self.ml_ds[BuyBakWines.goog],
            'y': self.ml_y[BuyBakWines.goog],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.goog]) * 0.8)
        print('------------ goog train_size --------')
        print(train_size)
        X_train[BuyBakWines.goog], y_train[BuyBakWines.goog] = X[BuyBakWines.goog][:train_size], y[BuyBakWines.goog][:train_size]

        #######
        # AMZN
        #######
        lf[BuyBakWines.AMZN] = df_ohlcv[BuyBakWines.AMZN][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.AMZN] = np.array(lf[BuyBakWines.AMZN]).tolist()
        y[BuyBakWines.AMZN] = np.array(df_ohlcv[BuyBakWines.AMZN]['ema']).tolist()

        # let's drop the last
        AMZN_len = len(y[BuyBakWines.AMZN])
        if AMZN_len % 2 == 1:
            y[BuyBakWines.AMZN].pop()
            AMZN_len = AMZN_len - 1

        self.ml_id[BuyBakWines.AMZN] = np.repeat(['id_00', 'id_01'], AMZN_len/2)
        self.ml_ds[BuyBakWines.AMZN] = pd.date_range('2000-01-01', periods=AMZN_len, freq='D')
        self.ml_y[BuyBakWines.AMZN] = y[BuyBakWines.AMZN] #df_ohlcv[BuyBakWines.AMZN]['ema']

        print(f'[AMZN] len {len(y[BuyBakWines.AMZN])}, {len(self.ml_id[BuyBakWines.AMZN])} , {len(self.ml_ds[BuyBakWines.AMZN])} , {len(self.ml_y[BuyBakWines.AMZN])} ')
        self.ml_series[BuyBakWines.AMZN] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.AMZN],
            'ds': self.ml_ds[BuyBakWines.AMZN],
            'y': self.ml_y[BuyBakWines.AMZN],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.AMZN]) * 0.8)
        print('------------ AMZN train_size --------')
        print(train_size)
        X_train[BuyBakWines.AMZN], y_train[BuyBakWines.AMZN] = X[BuyBakWines.AMZN][:train_size], y[BuyBakWines.AMZN][:train_size]

        #######
        # COST
        #######
        lf[BuyBakWines.COST] = df_ohlcv[BuyBakWines.COST][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.COST] = np.array(lf[BuyBakWines.COST]).tolist()
        y[BuyBakWines.COST] = np.array(df_ohlcv[BuyBakWines.COST]['ema']).tolist()


        # let's drop the last
        COST_len = len(y[BuyBakWines.COST])
        if COST_len % 2 == 1:
            y[BuyBakWines.COST].pop()
            COST_len = COST_len - 1

        self.ml_id[BuyBakWines.COST] = np.repeat(['id_00', 'id_01'], COST_len/2)
        self.ml_ds[BuyBakWines.COST] = pd.date_range('2000-01-01', periods=COST_len, freq='D')
        self.ml_y[BuyBakWines.COST] = y[BuyBakWines.COST] #df_ohlcv[BuyBakWines.COST]['ema']
        print(f'[COST] len {len(y[BuyBakWines.COST])}, {len(self.ml_id[BuyBakWines.COST])} , {len(self.ml_ds[BuyBakWines.COST])} , {len(self.ml_y[BuyBakWines.COST])} ')
        self.ml_series[BuyBakWines.COST] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.COST],
            'ds': self.ml_ds[BuyBakWines.COST],
            'y': self.ml_y[BuyBakWines.COST],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.COST]) * 0.8)
        print('------------ COST train_size --------')
        print(train_size)
        X_train[BuyBakWines.COST], y_train[BuyBakWines.COST] = X[BuyBakWines.COST][:train_size], y[BuyBakWines.COST][:train_size]

        #######
        # LLY
        #######
        lf[BuyBakWines.LLY] = df_ohlcv[BuyBakWines.LLY][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.LLY] = np.array(lf[BuyBakWines.LLY]).tolist()
        y[BuyBakWines.LLY] = np.array(df_ohlcv[BuyBakWines.LLY]['ema']).tolist()


        # let's drop the last
        LLY_len = len(y[BuyBakWines.LLY])
        if LLY_len % 2 == 1:
            y[BuyBakWines.LLY].pop()
            LLY_len = LLY_len - 1

        self.ml_id[BuyBakWines.LLY] = np.repeat(['id_00', 'id_01'], LLY_len/2)
        self.ml_ds[BuyBakWines.LLY] = pd.date_range('2000-01-01', periods=LLY_len, freq='D')
        self.ml_y[BuyBakWines.LLY] = y[BuyBakWines.LLY] #df_ohlcv[BuyBakWines.LLY]['ema']
        print(f'[LLY] len {len(y[BuyBakWines.LLY])}, {len(self.ml_id[BuyBakWines.LLY])} , {len(self.ml_ds[BuyBakWines.LLY])} , {len(self.ml_y[BuyBakWines.LLY])} ')
        self.ml_series[BuyBakWines.LLY] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.LLY],
            'ds': self.ml_ds[BuyBakWines.LLY],
            'y': self.ml_y[BuyBakWines.LLY],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.LLY]) * 0.8)
        print('------------ LLY train_size --------')
        print(train_size)
        X_train[BuyBakWines.LLY], y_train[BuyBakWines.LLY] = X[BuyBakWines.LLY][:train_size], y[BuyBakWines.LLY][:train_size]

        #######
        # META
        #######
        lf[BuyBakWines.META] = df_ohlcv[BuyBakWines.META][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.META] = np.array(lf[BuyBakWines.META]).tolist()
        y[BuyBakWines.META] = np.array(df_ohlcv[BuyBakWines.META]['ema']).tolist()


        # let's drop the last
        META_len = len(y[BuyBakWines.META])
        if META_len % 2 == 1:
            y[BuyBakWines.META].pop()
            META_len = META_len - 1

        self.ml_id[BuyBakWines.META] = np.repeat(['id_00', 'id_01'], META_len/2)
        self.ml_ds[BuyBakWines.META] = pd.date_range('2000-01-01', periods=META_len, freq='D')
        self.ml_y[BuyBakWines.META] = y[BuyBakWines.META] #df_ohlcv[BuyBakWines.META]['ema']
        print(f'[META] len {len(y[BuyBakWines.META])}, {len(self.ml_id[BuyBakWines.META])} , {len(self.ml_ds[BuyBakWines.META])} , {len(self.ml_y[BuyBakWines.META])} ')
        self.ml_series[BuyBakWines.META] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.META],
            'ds': self.ml_ds[BuyBakWines.META],
            'y': self.ml_y[BuyBakWines.META],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.META]) * 0.8)
        print('------------ META train_size --------')
        print(train_size)
        X_train[BuyBakWines.META], y_train[BuyBakWines.META] = X[BuyBakWines.META][:train_size], y[BuyBakWines.META][:train_size]

        #######
        # NVDA
        #######
        lf[BuyBakWines.NVDA] = df_ohlcv[BuyBakWines.NVDA][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.NVDA] = np.array(lf[BuyBakWines.NVDA]).tolist()
        y[BuyBakWines.NVDA] = np.array(df_ohlcv[BuyBakWines.NVDA]['ema']).tolist()


        # let's drop the last
        NVDA_len = len(y[BuyBakWines.NVDA])
        if NVDA_len % 2 == 1:
            y[BuyBakWines.NVDA].pop()
            NVDA_len = NVDA_len - 1

        self.ml_id[BuyBakWines.NVDA] = np.repeat(['id_00', 'id_01'], NVDA_len/2)
        self.ml_ds[BuyBakWines.NVDA] = pd.date_range('2000-01-01', periods=NVDA_len, freq='D')
        self.ml_y[BuyBakWines.NVDA] = y[BuyBakWines.NVDA] #df_ohlcv[BuyBakWines.NVDA]['ema']
        print(f'[NVDA] len {len(y[BuyBakWines.NVDA])}, {len(self.ml_id[BuyBakWines.NVDA])} , {len(self.ml_ds[BuyBakWines.NVDA])} , {len(self.ml_y[BuyBakWines.NVDA])} ')
        self.ml_series[BuyBakWines.NVDA] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.NVDA],
            'ds': self.ml_ds[BuyBakWines.NVDA],
            'y': self.ml_y[BuyBakWines.NVDA],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.NVDA]) * 0.8)
        print('------------ NVDA train_size --------')
        print(train_size)
        X_train[BuyBakWines.NVDA], y_train[BuyBakWines.NVDA] = X[BuyBakWines.NVDA][:train_size], y[BuyBakWines.NVDA][:train_size]

        #######
        # TSLA
        #######
        lf[BuyBakWines.TSLA] = df_ohlcv[BuyBakWines.TSLA][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.TSLA] = np.array(lf[BuyBakWines.TSLA]).tolist()
        y[BuyBakWines.TSLA] = np.array(df_ohlcv[BuyBakWines.TSLA]['ema']).tolist()


        # let's drop the last
        TSLA_len = len(y[BuyBakWines.TSLA])
        if TSLA_len % 2 == 1:
            y[BuyBakWines.TSLA].pop()
            TSLA_len = TSLA_len - 1

        self.ml_id[BuyBakWines.TSLA] = np.repeat(['id_00', 'id_01'], TSLA_len/2)
        self.ml_ds[BuyBakWines.TSLA] = pd.date_range('2000-01-01', periods=TSLA_len, freq='D')
        self.ml_y[BuyBakWines.TSLA] = y[BuyBakWines.TSLA] #df_ohlcv[BuyBakWines.TSLA]['ema']
        print(f'[TSLA] len {len(y[BuyBakWines.TSLA])}, {len(self.ml_id[BuyBakWines.TSLA])} , {len(self.ml_ds[BuyBakWines.TSLA])} , {len(self.ml_y[BuyBakWines.TSLA])} ')
        self.ml_series[BuyBakWines.TSLA] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.TSLA],
            'ds': self.ml_ds[BuyBakWines.TSLA],
            'y': self.ml_y[BuyBakWines.TSLA],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.TSLA]) * 0.8)
        print('------------ TSLA train_size --------')
        print(train_size)
        X_train[BuyBakWines.TSLA], y_train[BuyBakWines.TSLA] = X[BuyBakWines.TSLA][:train_size], y[BuyBakWines.TSLA][:train_size]

        #######
        # aapl
        #######
        lf[BuyBakWines.aapl] = df_ohlcv[BuyBakWines.aapl][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.aapl] = np.array(lf[BuyBakWines.aapl]).tolist()
        y[BuyBakWines.aapl] = np.array(df_ohlcv[BuyBakWines.aapl]['ema']).tolist()


        # let's drop the last
        aapl_len = len(y[BuyBakWines.aapl])
        if aapl_len % 2 == 1:
            y[BuyBakWines.aapl].pop()
            aapl_len = aapl_len - 1

        self.ml_id[BuyBakWines.aapl] = np.repeat(['id_00', 'id_01'], aapl_len/2)
        self.ml_ds[BuyBakWines.aapl] = pd.date_range('2000-01-01', periods=aapl_len, freq='D')
        self.ml_y[BuyBakWines.aapl] = y[BuyBakWines.aapl] #df_ohlcv[BuyBakWines.aapl]['ema']
        print(f'[aapl] len {len(y[BuyBakWines.aapl])}, {len(self.ml_id[BuyBakWines.aapl])} , {len(self.ml_ds[BuyBakWines.aapl])} , {len(self.ml_y[BuyBakWines.aapl])} ')
        self.ml_series[BuyBakWines.aapl] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.aapl],
            'ds': self.ml_ds[BuyBakWines.aapl],
            'y': self.ml_y[BuyBakWines.aapl],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.aapl]) * 0.8)
        print('------------ aapl train_size --------')
        print(train_size)
        X_train[BuyBakWines.aapl], y_train[BuyBakWines.aapl] = X[BuyBakWines.aapl][:train_size], y[BuyBakWines.aapl][:train_size]

        #######
        # MSFT
        #######
        lf[BuyBakWines.MSFT] = df_ohlcv[BuyBakWines.MSFT][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.MSFT] = np.array(lf[BuyBakWines.MSFT]).tolist()
        y[BuyBakWines.MSFT] = np.array(df_ohlcv[BuyBakWines.MSFT]['ema']).tolist()


        # let's drop the last
        MSFT_len = len(y[BuyBakWines.MSFT])
        if MSFT_len % 2 == 1:
            y[BuyBakWines.MSFT].pop()
            MSFT_len = MSFT_len - 1

        self.ml_id[BuyBakWines.MSFT] = np.repeat(['id_00', 'id_01'], MSFT_len/2)
        self.ml_ds[BuyBakWines.MSFT] = pd.date_range('2000-01-01', periods=MSFT_len, freq='D')
        self.ml_y[BuyBakWines.MSFT] = y[BuyBakWines.MSFT] #df_ohlcv[BuyBakWines.MSFT]['ema']
        print(f'[MSFT] len {len(y[BuyBakWines.MSFT])}, {len(self.ml_id[BuyBakWines.MSFT])} , {len(self.ml_ds[BuyBakWines.MSFT])} , {len(self.ml_y[BuyBakWines.MSFT])} ')
        self.ml_series[BuyBakWines.MSFT] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.MSFT],
            'ds': self.ml_ds[BuyBakWines.MSFT],
            'y': self.ml_y[BuyBakWines.MSFT],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.MSFT]) * 0.8)
        print('------------ MSFT train_size --------')
        print(train_size)
        X_train[BuyBakWines.MSFT], y_train[BuyBakWines.MSFT] = X[BuyBakWines.MSFT][:train_size], y[BuyBakWines.MSFT][:train_size]

        #######################
        # common to all wines
        #######################
        ml_models = [
            lgb.LGBMRegressor(random_state=0, verbosity=-1),
            # Add other models here if needed, e.g., LinearRegression()
        ]

        ###################################################################
        # Instantiate goog MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.goog] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.goog].fit(self.ml_series[BuyBakWines.goog])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.goog])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.goog])

        # Create and train the XGBoost model
        self.model[BuyBakWines.goog] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.goog].fit(X_train[BuyBakWines.goog], y_train[BuyBakWines.goog])
        print(self.model[BuyBakWines.goog])
        print('------ init model done --------')

        ###################################################################
        # Instantiate AMZN MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.AMZN] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.AMZN].fit(self.ml_series[BuyBakWines.AMZN])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.AMZN])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.AMZN])

        # Create and train the XGBoost model
        self.model[BuyBakWines.AMZN] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.AMZN].fit(X_train[BuyBakWines.AMZN], y_train[BuyBakWines.AMZN])
        print(self.model[BuyBakWines.AMZN])
        print('------ init model done --------')

        ###################################################################
        # Instantiate COST MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.COST] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.COST].fit(self.ml_series[BuyBakWines.COST])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.COST])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.COST])

        # Create and train the XGBoost model
        self.model[BuyBakWines.COST] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.COST].fit(X_train[BuyBakWines.COST], y_train[BuyBakWines.COST])
        print(self.model[BuyBakWines.COST])
        print('------ init model done --------')

        ###################################################################
        # Instantiate LLY MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.LLY] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.LLY].fit(self.ml_series[BuyBakWines.LLY])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.LLY])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.LLY])

        # Create and train the XGBoost model
        self.model[BuyBakWines.LLY] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.LLY].fit(X_train[BuyBakWines.LLY], y_train[BuyBakWines.LLY])
        print(self.model[BuyBakWines.LLY])
        print('------ init model done --------')

        ###################################################################
        # Instantiate META MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.META] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.META].fit(self.ml_series[BuyBakWines.META])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.META])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.META])

        # Create and train the XGBoost model
        self.model[BuyBakWines.META] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.META].fit(X_train[BuyBakWines.META], y_train[BuyBakWines.META])
        print(self.model[BuyBakWines.META])
        print('------ init model done --------')

        ###################################################################
        # Instantiate NVDA MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.NVDA] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.NVDA].fit(self.ml_series[BuyBakWines.NVDA])

        print('------------ NVDA X_train --------')
        print(X_train[BuyBakWines.NVDA])
        print('------------ NVDA y_train --------')
        print(y_train[BuyBakWines.NVDA])

        # Create and train the XGBoost model
        # self.model[BuyBakWines.NVDA] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        # self.model[BuyBakWines.NVDA].fit(X_train[BuyBakWines.NVDA], y_train[BuyBakWines.NVDA])
        # print(self.model[BuyBakWines.NVDA])
        # print('------ NVDA init model done --------')

        ###################################################################
        # Instantiate TSLA MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.TSLA] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.TSLA].fit(self.ml_series[BuyBakWines.TSLA])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.TSLA])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.TSLA])

        # Create and train the XGBoost model
        self.model[BuyBakWines.TSLA] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.TSLA].fit(X_train[BuyBakWines.TSLA], y_train[BuyBakWines.TSLA])
        print(self.model[BuyBakWines.TSLA])
        print('------ init model done --------')

        ###################################################################
        # Instantiate aapl MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.aapl] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.aapl].fit(self.ml_series[BuyBakWines.aapl])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.aapl])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.aapl])

        # Create and train the XGBoost model
        self.model[BuyBakWines.aapl] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.aapl].fit(X_train[BuyBakWines.aapl], y_train[BuyBakWines.aapl])
        print(self.model[BuyBakWines.aapl])
        print('------ init model done --------')

        ###################################################################
        # Instantiate MSFT MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.MSFT] = MLForecast(
            models=ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster[BuyBakWines.MSFT].fit(self.ml_series[BuyBakWines.MSFT])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.MSFT])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.MSFT])

        # Create and train the XGBoost model
        self.model[BuyBakWines.MSFT] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.MSFT].fit(X_train[BuyBakWines.MSFT], y_train[BuyBakWines.MSFT])
        print(self.model[BuyBakWines.MSFT])
        print('------ init model done --------')


    def extract_bbk_time_series_from_prompt(self, input: str) -> []:
        """Extract the BuyBakTimeSeries structure from the input """
        print("Inside extract_bbk_time_series_from_prompt")
        print(input)
        llm = OpenAI("gpt-4o-mini")
        sllm = llm.as_structured_llm(BuyBakTimeSeries)
        response = sllm.complete(input)
        print(response)
        return response.text


    def get_mean_squared_error(self) -> float:
        """Get the MSE from the live prediction ."""
        return self.mse[BuyBakWines.goog]

    def buybak_model_predict(self, argin: str) -> []:
        """Fit the model initialized (XGBoost) with the X_live data, calcumate the MSE with y_live, and return the y_pred_live"""

        # let's try to get the data, it could be in diff formats
        data = {}
        df_data = []

        try:
            print(argin)
            print('-------- argin above -------')
            import json
            data = json.loads(argin)
            print(data)
        except:
            return "Exception thrown parsing argin JSON"

        try:
            print('-------- input above -------')
            if "line_items" in  argin:
                print(f'-------- line_items')
                df_data = pd.DataFrame(data["line_items"])
            elif "ohlcv" in argin:
                print(f'-------- ohlcv')
                df_data = pd.DataFrame(data["ohlcv"])
            elif "data" in argin:
                print(f'-------- data')
                df_data = pd.DataFrame(data["data"])
            else:
                return "Could not find data in argin JSON"

            print(df_data)
            print(f'-------- df_data above ------- {type(df_data)}')
            lf = df_data[['open', 'high', 'low', 'close']]
            print(lf)
            print('-------- lf above -------')
            X_live = np.array(lf).tolist()
            print(X_live)
            print('-------- X_live above -------')
            y_live = lf['close']
            print(y_live)
            print('-------- y_live above -------')
            y_pred_live = self.model[BuyBakWines.goog].predict(X_live)
            print('---------- y_pred_live  -------------')
            print(y_pred_live)
            self.mse[BuyBakWines.goog] = mean_squared_error(y_live, y_pred_live)
            print(f'---------- self.mse  {self.mse[BuyBakWines.goog]}')
            return y_pred_live
        except:
            return "Exception thrown parsing argin JSON"

    def buybak_model_forecast(self, index: int, argin: str) -> []:
        """MLForecaster: Use the ml_forecaster model. 'index' defines which ml_forecaster to use based on the BuyBakWines IntEnum range(0-8). Predict the next N values, based on the 'argin' input, and return."""

        try:
            print(f'buybak_model_forecast({index}, {argin})')
            self.ml_predict[BuyBakWines.MSFT] = self.ml_forecaster[BuyBakWines.MSFT].predict(15)
            print('---------- ml_predict  -------------')
            print(self.ml_predict[BuyBakWines.MSFT])
            filtered_df = self.ml_predict[BuyBakWines.MSFT][(self.ml_predict[BuyBakWines.MSFT]['unique_id'] == "id_01")]
            filtered_df = filtered_df["LGBMRegressor"]
            return np.array(filtered_df).tolist()
        except:
            return "Exception thrown in ml_forecaster"

