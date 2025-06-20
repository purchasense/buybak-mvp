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
class BuyBakGermanLineItemAndIndicators(BaseModel):
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

class BuyBakGermanTimeSeries(BaseModel):
    """A struct to hold the line_items of OHLCV and  market indicators"""
    name: str = None
    line_items: List[BuyBakGermanLineItemAndIndicators] = Field(description="Line Items shipped with the indicators as response")

class BuyBakGermanWines(IntEnum):
    Ahr             = 0
    Baden           = 1
    Franconia       = 2
    Hessische       = 3
    Kaiserstuhl     = 4
    Mosel           = 5
    Rheinhessen     = 6
    German          = 7

class BuyBakGermanTimeSeriesToolSpec(BaseToolSpec):
    """BuyBak Apt Booking Tool spec."""

    spec_functions = [
        "extract_buybak_german_time_series_from_prompt",
        "get_buybak_german_mean_squared_error",
        "get_buybak_german_wines_enum",
        "buybak_german_model_forecast",
    ]
    mse                         = []
    storage_dir                 = "./storage-buybak-time-series"

    # XGBoost Regressor Model
    model           = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]

    # MLForecast Regressor Model
    ml_id           = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
    ml_ds           = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
    ml_y            = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
    ml_series       = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
    ml_predict      = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
    ml_forecaster   = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]



    def __init__(self):
        print('BBK: Initializing Time Series')

        #####################
        # reverse_Baden.cv
        #####################
        df_ohlcv = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
        X = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
        y = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
        lf = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
        X_train = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]
        y_train = [ BuyBakGermanWines.Ahr, BuyBakGermanWines.Baden, BuyBakGermanWines.Franconia, BuyBakGermanWines.Hessische, BuyBakGermanWines.Kaiserstuhl, BuyBakGermanWines.Mosel, BuyBakGermanWines.Rheinhessen, BuyBakGermanWines.German]

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Baden] = pd.read_csv("./TradesCSV/reverse_Baden.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Baden]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Baden]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Baden]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Baden]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Baden]['sma'] = df_ohlcv[BuyBakGermanWines.Baden]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Baden]['ema'] = df_ohlcv[BuyBakGermanWines.Baden]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Baden]['volatility'] = df_ohlcv[BuyBakGermanWines.Baden]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Baden].dropna(inplace=True)

        #####################
        # reverse_Franconia.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Franconia] = pd.read_csv("./TradesCSV/reverse_Franconia.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Franconia]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Franconia]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Franconia]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Franconia]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Franconia]['sma'] = df_ohlcv[BuyBakGermanWines.Franconia]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Franconia]['ema'] = df_ohlcv[BuyBakGermanWines.Franconia]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Franconia]['volatility'] = df_ohlcv[BuyBakGermanWines.Franconia]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Franconia].dropna(inplace=True)

        #####################
        # reverse_Hessische.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Hessische] = pd.read_csv("./TradesCSV/reverse_Hessische.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Hessische]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Hessische]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Hessische]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Hessische]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Hessische]['sma'] = df_ohlcv[BuyBakGermanWines.Hessische]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Hessische]['ema'] = df_ohlcv[BuyBakGermanWines.Hessische]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Hessische]['volatility'] = df_ohlcv[BuyBakGermanWines.Hessische]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Hessische].dropna(inplace=True)

        #####################
        # reverse_Kaiserstuhl.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl] = pd.read_csv("./TradesCSV/reverse_Kaiserstuhl.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['sma'] = df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['ema'] = df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['volatility'] = df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Kaiserstuhl].dropna(inplace=True)

        #####################
        # reverse_Mosel.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Mosel] = pd.read_csv("./TradesCSV/reverse_Mosel.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Mosel]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Mosel]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Mosel]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Mosel]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Mosel]['sma'] = df_ohlcv[BuyBakGermanWines.Mosel]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Mosel]['ema'] = df_ohlcv[BuyBakGermanWines.Mosel]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Mosel]['volatility'] = df_ohlcv[BuyBakGermanWines.Mosel]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Mosel].dropna(inplace=True)

        #####################
        # reverse_Rheinhessen.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Rheinhessen] = pd.read_csv("./TradesCSV/reverse_Rheinhessen.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Rheinhessen]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Rheinhessen]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Rheinhessen]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Rheinhessen]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Rheinhessen]['sma'] = df_ohlcv[BuyBakGermanWines.Rheinhessen]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Rheinhessen]['ema'] = df_ohlcv[BuyBakGermanWines.Rheinhessen]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Rheinhessen]['volatility'] = df_ohlcv[BuyBakGermanWines.Rheinhessen]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Rheinhessen].dropna(inplace=True)

        #####################
        # reverse_German.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.German] = pd.read_csv("./TradesCSV/reverse_German.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.German]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.German]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.German]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.German]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.German]['sma'] = df_ohlcv[BuyBakGermanWines.German]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.German]['ema'] = df_ohlcv[BuyBakGermanWines.German]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.German]['volatility'] = df_ohlcv[BuyBakGermanWines.German]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.German].dropna(inplace=True)

        #####################
        # reverse_Ahr.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakGermanWines.Ahr] = pd.read_csv("./TradesCSV/reverse_Ahr.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakGermanWines.Ahr]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakGermanWines.Ahr]['Close']).rsi()
        df_ohlcv[BuyBakGermanWines.Ahr]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakGermanWines.Ahr]['Close']).macd()
        df_ohlcv[BuyBakGermanWines.Ahr]['sma'] = df_ohlcv[BuyBakGermanWines.Ahr]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakGermanWines.Ahr]['ema'] = df_ohlcv[BuyBakGermanWines.Ahr]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakGermanWines.Ahr]['volatility'] = df_ohlcv[BuyBakGermanWines.Ahr]['Close'].rolling(14).std()
        df_ohlcv[BuyBakGermanWines.Ahr].dropna(inplace=True)

        ##############################
        # prepare X y
        ##############################

        #######
        # Baden
        #######
        lf[BuyBakGermanWines.Baden] = df_ohlcv[BuyBakGermanWines.Baden][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Baden] = np.array(lf[BuyBakGermanWines.Baden]).tolist()
        y[BuyBakGermanWines.Baden] = np.array(df_ohlcv[BuyBakGermanWines.Baden]['ema']).tolist()

        # let's drop the last
        Baden_len = len(y[BuyBakGermanWines.Baden])
        if Baden_len % 2 == 1:
            y[BuyBakGermanWines.Baden].pop()
            Baden_len = Baden_len - 1

        self.ml_id[BuyBakGermanWines.Baden] = np.repeat(['id_00', 'id_01'], Baden_len/2)
        self.ml_ds[BuyBakGermanWines.Baden] = pd.date_range('2000-01-01', periods=Baden_len, freq='D')
        self.ml_y[BuyBakGermanWines.Baden] = df_ohlcv[BuyBakGermanWines.Baden]['ema']
        print(f'[Baden] len {len(y[BuyBakGermanWines.Baden])}, {len(self.ml_id[BuyBakGermanWines.Baden])} , {len(self.ml_ds[BuyBakGermanWines.Baden])} , {len(self.ml_y[BuyBakGermanWines.Baden])} ')
        self.ml_series[BuyBakGermanWines.Baden] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Baden],
            'ds': self.ml_ds[BuyBakGermanWines.Baden],
            'y': self.ml_y[BuyBakGermanWines.Baden],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Baden]) * 0.8)
        print('------------ Baden train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Baden], y_train[BuyBakGermanWines.Baden] = X[BuyBakGermanWines.Baden][:train_size], y[BuyBakGermanWines.Baden][:train_size]

        #######
        # Franconia
        #######
        lf[BuyBakGermanWines.Franconia] = df_ohlcv[BuyBakGermanWines.Franconia][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Franconia] = np.array(lf[BuyBakGermanWines.Franconia]).tolist()
        y[BuyBakGermanWines.Franconia] = np.array(df_ohlcv[BuyBakGermanWines.Franconia]['ema']).tolist()

        # let's drop the last
        Franconia_len = len(y[BuyBakGermanWines.Franconia])
        if Franconia_len % 2 == 1:
            y[BuyBakGermanWines.Franconia].pop()
            Franconia_len = Franconia_len - 1

        self.ml_id[BuyBakGermanWines.Franconia] = np.repeat(['id_00', 'id_01'], Franconia_len/2)
        self.ml_ds[BuyBakGermanWines.Franconia] = pd.date_range('2000-01-01', periods=Franconia_len, freq='D')
        self.ml_y[BuyBakGermanWines.Franconia] = y[BuyBakGermanWines.Franconia] #df_ohlcv[BuyBakGermanWines.Franconia]['ema']

        print(f'[Franconia] len {len(y[BuyBakGermanWines.Franconia])}, {len(self.ml_id[BuyBakGermanWines.Franconia])} , {len(self.ml_ds[BuyBakGermanWines.Franconia])} , {len(self.ml_y[BuyBakGermanWines.Franconia])} ')
        self.ml_series[BuyBakGermanWines.Franconia] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Franconia],
            'ds': self.ml_ds[BuyBakGermanWines.Franconia],
            'y': self.ml_y[BuyBakGermanWines.Franconia],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Franconia]) * 0.8)
        print('------------ Franconia train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Franconia], y_train[BuyBakGermanWines.Franconia] = X[BuyBakGermanWines.Franconia][:train_size], y[BuyBakGermanWines.Franconia][:train_size]

        #######
        # Hessische
        #######
        lf[BuyBakGermanWines.Hessische] = df_ohlcv[BuyBakGermanWines.Hessische][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Hessische] = np.array(lf[BuyBakGermanWines.Hessische]).tolist()
        y[BuyBakGermanWines.Hessische] = np.array(df_ohlcv[BuyBakGermanWines.Hessische]['ema']).tolist()


        # let's drop the last
        Hessische_len = len(y[BuyBakGermanWines.Hessische])
        if Hessische_len % 2 == 1:
            y[BuyBakGermanWines.Hessische].pop()
            Hessische_len = Hessische_len - 1

        self.ml_id[BuyBakGermanWines.Hessische] = np.repeat(['id_00', 'id_01'], Hessische_len/2)
        self.ml_ds[BuyBakGermanWines.Hessische] = pd.date_range('2000-01-01', periods=Hessische_len, freq='D')
        self.ml_y[BuyBakGermanWines.Hessische] = y[BuyBakGermanWines.Hessische] #df_ohlcv[BuyBakGermanWines.Hessische]['ema']
        print(f'[Hessische] len {len(y[BuyBakGermanWines.Hessische])}, {len(self.ml_id[BuyBakGermanWines.Hessische])} , {len(self.ml_ds[BuyBakGermanWines.Hessische])} , {len(self.ml_y[BuyBakGermanWines.Hessische])} ')
        self.ml_series[BuyBakGermanWines.Hessische] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Hessische],
            'ds': self.ml_ds[BuyBakGermanWines.Hessische],
            'y': self.ml_y[BuyBakGermanWines.Hessische],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Hessische]) * 0.8)
        print('------------ Hessische train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Hessische], y_train[BuyBakGermanWines.Hessische] = X[BuyBakGermanWines.Hessische][:train_size], y[BuyBakGermanWines.Hessische][:train_size]

        #######
        # Kaiserstuhl
        #######
        lf[BuyBakGermanWines.Kaiserstuhl] = df_ohlcv[BuyBakGermanWines.Kaiserstuhl][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Kaiserstuhl] = np.array(lf[BuyBakGermanWines.Kaiserstuhl]).tolist()
        y[BuyBakGermanWines.Kaiserstuhl] = np.array(df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['ema']).tolist()


        # let's drop the last
        Kaiserstuhl_len = len(y[BuyBakGermanWines.Kaiserstuhl])
        if Kaiserstuhl_len % 2 == 1:
            y[BuyBakGermanWines.Kaiserstuhl].pop()
            Kaiserstuhl_len = Kaiserstuhl_len - 1

        self.ml_id[BuyBakGermanWines.Kaiserstuhl] = np.repeat(['id_00', 'id_01'], Kaiserstuhl_len/2)
        self.ml_ds[BuyBakGermanWines.Kaiserstuhl] = pd.date_range('2000-01-01', periods=Kaiserstuhl_len, freq='D')
        self.ml_y[BuyBakGermanWines.Kaiserstuhl] = y[BuyBakGermanWines.Kaiserstuhl] #df_ohlcv[BuyBakGermanWines.Kaiserstuhl]['ema']
        print(f'[Kaiserstuhl] len {len(y[BuyBakGermanWines.Kaiserstuhl])}, {len(self.ml_id[BuyBakGermanWines.Kaiserstuhl])} , {len(self.ml_ds[BuyBakGermanWines.Kaiserstuhl])} , {len(self.ml_y[BuyBakGermanWines.Kaiserstuhl])} ')
        self.ml_series[BuyBakGermanWines.Kaiserstuhl] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Kaiserstuhl],
            'ds': self.ml_ds[BuyBakGermanWines.Kaiserstuhl],
            'y': self.ml_y[BuyBakGermanWines.Kaiserstuhl],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Kaiserstuhl]) * 0.8)
        print('------------ Kaiserstuhl train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Kaiserstuhl], y_train[BuyBakGermanWines.Kaiserstuhl] = X[BuyBakGermanWines.Kaiserstuhl][:train_size], y[BuyBakGermanWines.Kaiserstuhl][:train_size]

        #######
        # Mosel
        #######
        lf[BuyBakGermanWines.Mosel] = df_ohlcv[BuyBakGermanWines.Mosel][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Mosel] = np.array(lf[BuyBakGermanWines.Mosel]).tolist()
        y[BuyBakGermanWines.Mosel] = np.array(df_ohlcv[BuyBakGermanWines.Mosel]['ema']).tolist()


        # let's drop the last
        Mosel_len = len(y[BuyBakGermanWines.Mosel])
        if Mosel_len % 2 == 1:
            y[BuyBakGermanWines.Mosel].pop()
            Mosel_len = Mosel_len - 1

        self.ml_id[BuyBakGermanWines.Mosel] = np.repeat(['id_00', 'id_01'], Mosel_len/2)
        self.ml_ds[BuyBakGermanWines.Mosel] = pd.date_range('2000-01-01', periods=Mosel_len, freq='D')
        self.ml_y[BuyBakGermanWines.Mosel] = y[BuyBakGermanWines.Mosel] #df_ohlcv[BuyBakGermanWines.Mosel]['ema']
        print(f'[Mosel] len {len(y[BuyBakGermanWines.Mosel])}, {len(self.ml_id[BuyBakGermanWines.Mosel])} , {len(self.ml_ds[BuyBakGermanWines.Mosel])} , {len(self.ml_y[BuyBakGermanWines.Mosel])} ')
        self.ml_series[BuyBakGermanWines.Mosel] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Mosel],
            'ds': self.ml_ds[BuyBakGermanWines.Mosel],
            'y': self.ml_y[BuyBakGermanWines.Mosel],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Mosel]) * 0.8)
        print('------------ Mosel train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Mosel], y_train[BuyBakGermanWines.Mosel] = X[BuyBakGermanWines.Mosel][:train_size], y[BuyBakGermanWines.Mosel][:train_size]

        #######
        # Rheinhessen
        #######
        lf[BuyBakGermanWines.Rheinhessen] = df_ohlcv[BuyBakGermanWines.Rheinhessen][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Rheinhessen] = np.array(lf[BuyBakGermanWines.Rheinhessen]).tolist()
        y[BuyBakGermanWines.Rheinhessen] = np.array(df_ohlcv[BuyBakGermanWines.Rheinhessen]['ema']).tolist()


        # let's drop the last
        Rheinhessen_len = len(y[BuyBakGermanWines.Rheinhessen])
        if Rheinhessen_len % 2 == 1:
            y[BuyBakGermanWines.Rheinhessen].pop()
            Rheinhessen_len = Rheinhessen_len - 1

        self.ml_id[BuyBakGermanWines.Rheinhessen] = np.repeat(['id_00', 'id_01'], Rheinhessen_len/2)
        self.ml_ds[BuyBakGermanWines.Rheinhessen] = pd.date_range('2000-01-01', periods=Rheinhessen_len, freq='D')
        self.ml_y[BuyBakGermanWines.Rheinhessen] = y[BuyBakGermanWines.Rheinhessen] #df_ohlcv[BuyBakGermanWines.Rheinhessen]['ema']
        print(f'[Rheinhessen] len {len(y[BuyBakGermanWines.Rheinhessen])}, {len(self.ml_id[BuyBakGermanWines.Rheinhessen])} , {len(self.ml_ds[BuyBakGermanWines.Rheinhessen])} , {len(self.ml_y[BuyBakGermanWines.Rheinhessen])} ')
        self.ml_series[BuyBakGermanWines.Rheinhessen] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Rheinhessen],
            'ds': self.ml_ds[BuyBakGermanWines.Rheinhessen],
            'y': self.ml_y[BuyBakGermanWines.Rheinhessen],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Rheinhessen]) * 0.8)
        print('------------ Rheinhessen train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Rheinhessen], y_train[BuyBakGermanWines.Rheinhessen] = X[BuyBakGermanWines.Rheinhessen][:train_size], y[BuyBakGermanWines.Rheinhessen][:train_size]

        #######
        # German
        #######
        lf[BuyBakGermanWines.German] = df_ohlcv[BuyBakGermanWines.German][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.German] = np.array(lf[BuyBakGermanWines.German]).tolist()
        y[BuyBakGermanWines.German] = np.array(df_ohlcv[BuyBakGermanWines.German]['ema']).tolist()


        # let's drop the last
        German_len = len(y[BuyBakGermanWines.German])
        if German_len % 2 == 1:
            y[BuyBakGermanWines.German].pop()
            German_len = German_len - 1

        self.ml_id[BuyBakGermanWines.German] = np.repeat(['id_00', 'id_01'], German_len/2)
        self.ml_ds[BuyBakGermanWines.German] = pd.date_range('2000-01-01', periods=German_len, freq='D')
        self.ml_y[BuyBakGermanWines.German] = y[BuyBakGermanWines.German] #df_ohlcv[BuyBakGermanWines.German]['ema']
        print(f'[German] len {len(y[BuyBakGermanWines.German])}, {len(self.ml_id[BuyBakGermanWines.German])} , {len(self.ml_ds[BuyBakGermanWines.German])} , {len(self.ml_y[BuyBakGermanWines.German])} ')
        self.ml_series[BuyBakGermanWines.German] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.German],
            'ds': self.ml_ds[BuyBakGermanWines.German],
            'y': self.ml_y[BuyBakGermanWines.German],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.German]) * 0.8)
        print('------------ German train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.German], y_train[BuyBakGermanWines.German] = X[BuyBakGermanWines.German][:train_size], y[BuyBakGermanWines.German][:train_size]


        #######
        # Ahr
        #######
        lf[BuyBakGermanWines.Ahr] = df_ohlcv[BuyBakGermanWines.Ahr][['Open', 'High', 'Low', 'Close']]
        X[BuyBakGermanWines.Ahr] = np.array(lf[BuyBakGermanWines.Ahr]).tolist()
        y[BuyBakGermanWines.Ahr] = np.array(df_ohlcv[BuyBakGermanWines.Ahr]['ema']).tolist()


        # let's drop the last
        Ahr = len(y[BuyBakGermanWines.Ahr])
        if Ahr % 2 == 1:
            y[BuyBakGermanWines.Ahr].pop()
            Ahr = Ahr - 1

        self.ml_id[BuyBakGermanWines.Ahr] = np.repeat(['id_00', 'id_01'], Ahr/2)
        self.ml_ds[BuyBakGermanWines.Ahr] = pd.date_range('2000-01-01', periods=Ahr, freq='D')
        self.ml_y[BuyBakGermanWines.Ahr] = y[BuyBakGermanWines.Ahr] #df_ohlcv[BuyBakGermanWines.Ahr]['ema']
        print(f'[Ahr] len {len(y[BuyBakGermanWines.Ahr])}, {len(self.ml_id[BuyBakGermanWines.Ahr])} , {len(self.ml_ds[BuyBakGermanWines.Ahr])} , {len(self.ml_y[BuyBakGermanWines.Ahr])} ')
        self.ml_series[BuyBakGermanWines.Ahr] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakGermanWines.Ahr],
            'ds': self.ml_ds[BuyBakGermanWines.Ahr],
            'y': self.ml_y[BuyBakGermanWines.Ahr],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakGermanWines.Ahr]) * 0.8)
        print('------------ Ahr train_size --------')
        print(train_size)
        X_train[BuyBakGermanWines.Ahr], y_train[BuyBakGermanWines.Ahr] = X[BuyBakGermanWines.Ahr][:train_size], y[BuyBakGermanWines.Ahr][:train_size]

        #######################
        # common to all wines
        #######################
        ml_models = [
            lgb.LGBMRegressor(random_state=0, verbosity=-1),
            # Add other models here if needed, e.g., LinearRegression()
        ]

        ###################################################################
        # Instantiate Baden MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Baden] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Baden].fit(self.ml_series[BuyBakGermanWines.Baden])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Baden])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Baden])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Baden] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Baden].fit(X_train[BuyBakGermanWines.Baden], y_train[BuyBakGermanWines.Baden])
        print(self.model[BuyBakGermanWines.Baden])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Franconia MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Franconia] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Franconia].fit(self.ml_series[BuyBakGermanWines.Franconia])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Franconia])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Franconia])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Franconia] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Franconia].fit(X_train[BuyBakGermanWines.Franconia], y_train[BuyBakGermanWines.Franconia])
        print(self.model[BuyBakGermanWines.Franconia])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Hessische MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Hessische] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Hessische].fit(self.ml_series[BuyBakGermanWines.Hessische])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Hessische])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Hessische])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Hessische] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Hessische].fit(X_train[BuyBakGermanWines.Hessische], y_train[BuyBakGermanWines.Hessische])
        print(self.model[BuyBakGermanWines.Hessische])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Kaiserstuhl MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Kaiserstuhl] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Kaiserstuhl].fit(self.ml_series[BuyBakGermanWines.Kaiserstuhl])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Kaiserstuhl])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Kaiserstuhl])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Kaiserstuhl] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Kaiserstuhl].fit(X_train[BuyBakGermanWines.Kaiserstuhl], y_train[BuyBakGermanWines.Kaiserstuhl])
        print(self.model[BuyBakGermanWines.Kaiserstuhl])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Mosel MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Mosel] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Mosel].fit(self.ml_series[BuyBakGermanWines.Mosel])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Mosel])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Mosel])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Mosel] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Mosel].fit(X_train[BuyBakGermanWines.Mosel], y_train[BuyBakGermanWines.Mosel])
        print(self.model[BuyBakGermanWines.Mosel])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Rheinhessen MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Rheinhessen] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Rheinhessen].fit(self.ml_series[BuyBakGermanWines.Rheinhessen])

        print('------------ Rheinhessen X_train --------')
        print(X_train[BuyBakGermanWines.Rheinhessen])
        print('------------ Rheinhessen y_train --------')
        print(y_train[BuyBakGermanWines.Rheinhessen])

        # Create and train the XGBoost model
        # self.model[BuyBakGermanWines.Rheinhessen] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        # self.model[BuyBakGermanWines.Rheinhessen].fit(X_train[BuyBakGermanWines.Rheinhessen], y_train[BuyBakGermanWines.Rheinhessen])
        # print(self.model[BuyBakGermanWines.Rheinhessen])
        # print('------ Rheinhessen init model done --------')

        ###################################################################
        # Instantiate German MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.German] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.German].fit(self.ml_series[BuyBakGermanWines.German])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.German])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.German])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.German] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.German].fit(X_train[BuyBakGermanWines.German], y_train[BuyBakGermanWines.German])
        print(self.model[BuyBakGermanWines.German])
        print('------ init model done --------')


        ###################################################################
        # Instantiate Ahr MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakGermanWines.Ahr] = MLForecast(
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
        self.ml_forecaster[BuyBakGermanWines.Ahr].fit(self.ml_series[BuyBakGermanWines.Ahr])

        # print('------------ X_train --------')
        # print(X_train[BuyBakGermanWines.Ahr])
        # print('------------ y_train --------')
        # print(y_train[BuyBakGermanWines.Ahr])

        # Create and train the XGBoost model
        self.model[BuyBakGermanWines.Ahr] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakGermanWines.Ahr].fit(X_train[BuyBakGermanWines.Ahr], y_train[BuyBakGermanWines.Ahr])
        print(self.model[BuyBakGermanWines.Ahr])
        print('------ init model done --------')


    def extract_buybak_german_time_series_from_prompt(self, input: str) -> []:
        """Extract the BuyBakGermanTimeSeries structure from the input """
        print("Inside extract_bbk_time_series_from_prompt")
        print(input)
        llm = OpenAI("gpt-4.1")
        sllm = llm.as_structured_llm(BuyBakGermanTimeSeries)
        response = sllm.complete(input)
        print(response)
        return response.text


    def get_buybak_german_mean_squared_error(self) -> float:
        """Get the MSE from the live prediction ."""
        return self.mse[BuyBakGermanWines.Baden]

    def buybak_german_model_predict(self, argin: str) -> []:
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
            y_pred_live = self.model[BuyBakGermanWines.Baden].predict(X_live)
            print('---------- y_pred_live  -------------')
            print(y_pred_live)
            self.mse[BuyBakGermanWines.Baden] = mean_squared_error(y_live, y_pred_live)
            print(f'---------- self.mse  {self.mse[BuyBakGermanWines.Baden]}')
            return y_pred_live
        except:
            return "Exception thrown parsing argin JSON"

    def buybak_german_model_forecast(self, index: int, argin: int) -> []:
        """MLForecaster: Use the ml_forecaster model. 'index' defines which ml_forecaster to use based on the BuyBakGermanWines IntEnum range(0-8). Predict the next N values, based on the 'argin' input, and return."""

        try:
            print(f'buybak_model_forecast({index}, {argin})')
            self.ml_predict[index] = self.ml_forecaster[index].predict(argin)
            print('---------- ml_predict  -------------')
            print(self.ml_predict[index])
            filtered_df = self.ml_predict[index][(self.ml_predict[index]['unique_id'] == "id_01")]
            filtered_df = filtered_df["LGBMRegressor"]
            return np.array(filtered_df).tolist()
        except:
            return "Exception thrown in ml_forecaster"

    def get_buybak_german_wines_enum(self) -> str:
        return "{'WinesEnum': [ {'Ahr': 0}, {'Baden': 1}, {'Franconia': 2}, {'Hessische': 3}, {'Kaiserstuhl':  4}, {'Mosel': 5}, {'German': 7}, {'Rheinhessen': 6}]}"
