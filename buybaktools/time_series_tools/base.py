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
    Alsace              = 0
    Bordeaux            = 1
    Burgundy            = 2
    Champagne           = 3
    Corsican            = 4
    French              = 5
    SouthWestFrench     = 6
    Jura                = 7
    Savoy               = 8

class BuyBakTimeSeriesToolSpec(BaseToolSpec):
    """BuyBak Apt Booking Tool spec."""

    spec_functions = [
        "extract_bbk_time_series_from_prompt",
        "get_mean_squared_error",
        "get_buybak_wines_enum",
        "buybak_model_forecast",
    ]
    mse                         = []
    storage_dir                 = "./storage-buybak-time-series"

    # XGBoost Regressor Model
    model           = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]

    # MLForecast Regressor Model
    ml_id           = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
    ml_ds           = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
    ml_y            = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
    ml_series       = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
    ml_predict      = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
    ml_forecaster   = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]



    def __init__(self):
        print('BBK: Initializing Time Series')

        #####################
        # reverse_Alsace.cv
        #####################
        df_ohlcv = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
        X = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
        y = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
        lf = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
        X_train = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]
        y_train = [ BuyBakWines.Alsace, BuyBakWines.Bordeaux, BuyBakWines.Burgundy, BuyBakWines.Champagne, BuyBakWines.Corsican, BuyBakWines.SouthWestFrench, BuyBakWines.French, BuyBakWines.Jura, BuyBakWines.Savoy ]

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Alsace] = pd.read_csv("./TradesCSV/reverse_Alsace.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Alsace]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Alsace]['Close']).rsi()
        df_ohlcv[BuyBakWines.Alsace]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Alsace]['Close']).macd()
        df_ohlcv[BuyBakWines.Alsace]['sma'] = df_ohlcv[BuyBakWines.Alsace]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Alsace]['ema'] = df_ohlcv[BuyBakWines.Alsace]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Alsace]['volatility'] = df_ohlcv[BuyBakWines.Alsace]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Alsace].dropna(inplace=True)

        #####################
        # reverse_Bordeaux.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Bordeaux] = pd.read_csv("./TradesCSV/reverse_Bordeaux.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Bordeaux]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Bordeaux]['Close']).rsi()
        df_ohlcv[BuyBakWines.Bordeaux]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Bordeaux]['Close']).macd()
        df_ohlcv[BuyBakWines.Bordeaux]['sma'] = df_ohlcv[BuyBakWines.Bordeaux]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Bordeaux]['ema'] = df_ohlcv[BuyBakWines.Bordeaux]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Bordeaux]['volatility'] = df_ohlcv[BuyBakWines.Bordeaux]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Bordeaux].dropna(inplace=True)

        #####################
        # reverse_Burgundy.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Burgundy] = pd.read_csv("./TradesCSV/reverse_Burgundy.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Burgundy]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Burgundy]['Close']).rsi()
        df_ohlcv[BuyBakWines.Burgundy]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Burgundy]['Close']).macd()
        df_ohlcv[BuyBakWines.Burgundy]['sma'] = df_ohlcv[BuyBakWines.Burgundy]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Burgundy]['ema'] = df_ohlcv[BuyBakWines.Burgundy]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Burgundy]['volatility'] = df_ohlcv[BuyBakWines.Burgundy]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Burgundy].dropna(inplace=True)

        #####################
        # reverse_Champagne.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Champagne] = pd.read_csv("./TradesCSV/reverse_Champagne.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Champagne]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Champagne]['Close']).rsi()
        df_ohlcv[BuyBakWines.Champagne]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Champagne]['Close']).macd()
        df_ohlcv[BuyBakWines.Champagne]['sma'] = df_ohlcv[BuyBakWines.Champagne]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Champagne]['ema'] = df_ohlcv[BuyBakWines.Champagne]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Champagne]['volatility'] = df_ohlcv[BuyBakWines.Champagne]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Champagne].dropna(inplace=True)

        #####################
        # reverse_Corsican.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Corsican] = pd.read_csv("./TradesCSV/reverse_Corsican.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Corsican]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Corsican]['Close']).rsi()
        df_ohlcv[BuyBakWines.Corsican]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Corsican]['Close']).macd()
        df_ohlcv[BuyBakWines.Corsican]['sma'] = df_ohlcv[BuyBakWines.Corsican]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Corsican]['ema'] = df_ohlcv[BuyBakWines.Corsican]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Corsican]['volatility'] = df_ohlcv[BuyBakWines.Corsican]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Corsican].dropna(inplace=True)

        #####################
        # reverse_SouthWestFrench.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.SouthWestFrench] = pd.read_csv("./TradesCSV/reverse_SouthWestFrench.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.SouthWestFrench]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.SouthWestFrench]['Close']).rsi()
        df_ohlcv[BuyBakWines.SouthWestFrench]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.SouthWestFrench]['Close']).macd()
        df_ohlcv[BuyBakWines.SouthWestFrench]['sma'] = df_ohlcv[BuyBakWines.SouthWestFrench]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.SouthWestFrench]['ema'] = df_ohlcv[BuyBakWines.SouthWestFrench]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.SouthWestFrench]['volatility'] = df_ohlcv[BuyBakWines.SouthWestFrench]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.SouthWestFrench].dropna(inplace=True)

        #####################
        # reverse_French.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.French] = pd.read_csv("./TradesCSV/reverse_French.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.French]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.French]['Close']).rsi()
        df_ohlcv[BuyBakWines.French]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.French]['Close']).macd()
        df_ohlcv[BuyBakWines.French]['sma'] = df_ohlcv[BuyBakWines.French]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.French]['ema'] = df_ohlcv[BuyBakWines.French]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.French]['volatility'] = df_ohlcv[BuyBakWines.French]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.French].dropna(inplace=True)

        #####################
        # reverse_Jura.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Jura] = pd.read_csv("./TradesCSV/reverse_Jura.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Jura]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Jura]['Close']).rsi()
        df_ohlcv[BuyBakWines.Jura]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Jura]['Close']).macd()
        df_ohlcv[BuyBakWines.Jura]['sma'] = df_ohlcv[BuyBakWines.Jura]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Jura]['ema'] = df_ohlcv[BuyBakWines.Jura]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Jura]['volatility'] = df_ohlcv[BuyBakWines.Jura]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Jura].dropna(inplace=True)

        #####################
        # reverse_Savoy.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakWines.Savoy] = pd.read_csv("./TradesCSV/reverse_Savoy.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakWines.Savoy]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakWines.Savoy]['Close']).rsi()
        df_ohlcv[BuyBakWines.Savoy]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakWines.Savoy]['Close']).macd()
        df_ohlcv[BuyBakWines.Savoy]['sma'] = df_ohlcv[BuyBakWines.Savoy]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakWines.Savoy]['ema'] = df_ohlcv[BuyBakWines.Savoy]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakWines.Savoy]['volatility'] = df_ohlcv[BuyBakWines.Savoy]['Close'].rolling(14).std()
        df_ohlcv[BuyBakWines.Savoy].dropna(inplace=True)

        ##############################
        # prepare X y
        ##############################

        #######
        # Alsace
        #######
        lf[BuyBakWines.Alsace] = df_ohlcv[BuyBakWines.Alsace][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Alsace] = np.array(lf[BuyBakWines.Alsace]).tolist()
        y[BuyBakWines.Alsace] = np.array(df_ohlcv[BuyBakWines.Alsace]['ema']).tolist()

        # let's drop the last
        Alsace_len = len(y[BuyBakWines.Alsace])
        if Alsace_len % 2 == 1:
            y[BuyBakWines.Alsace].pop()
            Alsace_len = Alsace_len - 1

        self.ml_id[BuyBakWines.Alsace] = np.repeat(['id_00', 'id_01'], Alsace_len/2)
        self.ml_ds[BuyBakWines.Alsace] = pd.date_range('2000-01-01', periods=Alsace_len, freq='D')
        self.ml_y[BuyBakWines.Alsace] = df_ohlcv[BuyBakWines.Alsace]['ema']
        print(f'[Alsace] len {len(y[BuyBakWines.Alsace])}, {len(self.ml_id[BuyBakWines.Alsace])} , {len(self.ml_ds[BuyBakWines.Alsace])} , {len(self.ml_y[BuyBakWines.Alsace])} ')
        self.ml_series[BuyBakWines.Alsace] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Alsace],
            'ds': self.ml_ds[BuyBakWines.Alsace],
            'y': self.ml_y[BuyBakWines.Alsace],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Alsace]) * 0.8)
        print('------------ Alsace train_size --------')
        print(train_size)
        X_train[BuyBakWines.Alsace], y_train[BuyBakWines.Alsace] = X[BuyBakWines.Alsace][:train_size], y[BuyBakWines.Alsace][:train_size]

        #######
        # Bordeaux
        #######
        lf[BuyBakWines.Bordeaux] = df_ohlcv[BuyBakWines.Bordeaux][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Bordeaux] = np.array(lf[BuyBakWines.Bordeaux]).tolist()
        y[BuyBakWines.Bordeaux] = np.array(df_ohlcv[BuyBakWines.Bordeaux]['ema']).tolist()

        # let's drop the last
        Bordeaux_len = len(y[BuyBakWines.Bordeaux])
        if Bordeaux_len % 2 == 1:
            y[BuyBakWines.Bordeaux].pop()
            Bordeaux_len = Bordeaux_len - 1

        self.ml_id[BuyBakWines.Bordeaux] = np.repeat(['id_00', 'id_01'], Bordeaux_len/2)
        self.ml_ds[BuyBakWines.Bordeaux] = pd.date_range('2000-01-01', periods=Bordeaux_len, freq='D')
        self.ml_y[BuyBakWines.Bordeaux] = y[BuyBakWines.Bordeaux] #df_ohlcv[BuyBakWines.Bordeaux]['ema']

        print(f'[Bordeaux] len {len(y[BuyBakWines.Bordeaux])}, {len(self.ml_id[BuyBakWines.Bordeaux])} , {len(self.ml_ds[BuyBakWines.Bordeaux])} , {len(self.ml_y[BuyBakWines.Bordeaux])} ')
        self.ml_series[BuyBakWines.Bordeaux] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Bordeaux],
            'ds': self.ml_ds[BuyBakWines.Bordeaux],
            'y': self.ml_y[BuyBakWines.Bordeaux],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Bordeaux]) * 0.8)
        print('------------ Bordeaux train_size --------')
        print(train_size)
        X_train[BuyBakWines.Bordeaux], y_train[BuyBakWines.Bordeaux] = X[BuyBakWines.Bordeaux][:train_size], y[BuyBakWines.Bordeaux][:train_size]

        #######
        # Burgundy
        #######
        lf[BuyBakWines.Burgundy] = df_ohlcv[BuyBakWines.Burgundy][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Burgundy] = np.array(lf[BuyBakWines.Burgundy]).tolist()
        y[BuyBakWines.Burgundy] = np.array(df_ohlcv[BuyBakWines.Burgundy]['ema']).tolist()


        # let's drop the last
        Burgundy_len = len(y[BuyBakWines.Burgundy])
        if Burgundy_len % 2 == 1:
            y[BuyBakWines.Burgundy].pop()
            Burgundy_len = Burgundy_len - 1

        self.ml_id[BuyBakWines.Burgundy] = np.repeat(['id_00', 'id_01'], Burgundy_len/2)
        self.ml_ds[BuyBakWines.Burgundy] = pd.date_range('2000-01-01', periods=Burgundy_len, freq='D')
        self.ml_y[BuyBakWines.Burgundy] = y[BuyBakWines.Burgundy] #df_ohlcv[BuyBakWines.Burgundy]['ema']
        print(f'[Burgundy] len {len(y[BuyBakWines.Burgundy])}, {len(self.ml_id[BuyBakWines.Burgundy])} , {len(self.ml_ds[BuyBakWines.Burgundy])} , {len(self.ml_y[BuyBakWines.Burgundy])} ')
        self.ml_series[BuyBakWines.Burgundy] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Burgundy],
            'ds': self.ml_ds[BuyBakWines.Burgundy],
            'y': self.ml_y[BuyBakWines.Burgundy],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Burgundy]) * 0.8)
        print('------------ Burgundy train_size --------')
        print(train_size)
        X_train[BuyBakWines.Burgundy], y_train[BuyBakWines.Burgundy] = X[BuyBakWines.Burgundy][:train_size], y[BuyBakWines.Burgundy][:train_size]

        #######
        # Champagne
        #######
        lf[BuyBakWines.Champagne] = df_ohlcv[BuyBakWines.Champagne][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Champagne] = np.array(lf[BuyBakWines.Champagne]).tolist()
        y[BuyBakWines.Champagne] = np.array(df_ohlcv[BuyBakWines.Champagne]['ema']).tolist()


        # let's drop the last
        Champagne_len = len(y[BuyBakWines.Champagne])
        if Champagne_len % 2 == 1:
            y[BuyBakWines.Champagne].pop()
            Champagne_len = Champagne_len - 1

        self.ml_id[BuyBakWines.Champagne] = np.repeat(['id_00', 'id_01'], Champagne_len/2)
        self.ml_ds[BuyBakWines.Champagne] = pd.date_range('2000-01-01', periods=Champagne_len, freq='D')
        self.ml_y[BuyBakWines.Champagne] = y[BuyBakWines.Champagne] #df_ohlcv[BuyBakWines.Champagne]['ema']
        print(f'[Champagne] len {len(y[BuyBakWines.Champagne])}, {len(self.ml_id[BuyBakWines.Champagne])} , {len(self.ml_ds[BuyBakWines.Champagne])} , {len(self.ml_y[BuyBakWines.Champagne])} ')
        self.ml_series[BuyBakWines.Champagne] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Champagne],
            'ds': self.ml_ds[BuyBakWines.Champagne],
            'y': self.ml_y[BuyBakWines.Champagne],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Champagne]) * 0.8)
        print('------------ Champagne train_size --------')
        print(train_size)
        X_train[BuyBakWines.Champagne], y_train[BuyBakWines.Champagne] = X[BuyBakWines.Champagne][:train_size], y[BuyBakWines.Champagne][:train_size]

        #######
        # Corsican
        #######
        lf[BuyBakWines.Corsican] = df_ohlcv[BuyBakWines.Corsican][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Corsican] = np.array(lf[BuyBakWines.Corsican]).tolist()
        y[BuyBakWines.Corsican] = np.array(df_ohlcv[BuyBakWines.Corsican]['ema']).tolist()


        # let's drop the last
        Corsican_len = len(y[BuyBakWines.Corsican])
        if Corsican_len % 2 == 1:
            y[BuyBakWines.Corsican].pop()
            Corsican_len = Corsican_len - 1

        self.ml_id[BuyBakWines.Corsican] = np.repeat(['id_00', 'id_01'], Corsican_len/2)
        self.ml_ds[BuyBakWines.Corsican] = pd.date_range('2000-01-01', periods=Corsican_len, freq='D')
        self.ml_y[BuyBakWines.Corsican] = y[BuyBakWines.Corsican] #df_ohlcv[BuyBakWines.Corsican]['ema']
        print(f'[Corsican] len {len(y[BuyBakWines.Corsican])}, {len(self.ml_id[BuyBakWines.Corsican])} , {len(self.ml_ds[BuyBakWines.Corsican])} , {len(self.ml_y[BuyBakWines.Corsican])} ')
        self.ml_series[BuyBakWines.Corsican] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Corsican],
            'ds': self.ml_ds[BuyBakWines.Corsican],
            'y': self.ml_y[BuyBakWines.Corsican],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Corsican]) * 0.8)
        print('------------ Corsican train_size --------')
        print(train_size)
        X_train[BuyBakWines.Corsican], y_train[BuyBakWines.Corsican] = X[BuyBakWines.Corsican][:train_size], y[BuyBakWines.Corsican][:train_size]

        #######
        # SouthWestFrench
        #######
        lf[BuyBakWines.SouthWestFrench] = df_ohlcv[BuyBakWines.SouthWestFrench][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.SouthWestFrench] = np.array(lf[BuyBakWines.SouthWestFrench]).tolist()
        y[BuyBakWines.SouthWestFrench] = np.array(df_ohlcv[BuyBakWines.SouthWestFrench]['ema']).tolist()


        # let's drop the last
        SouthWestFrench_len = len(y[BuyBakWines.SouthWestFrench])
        if SouthWestFrench_len % 2 == 1:
            y[BuyBakWines.SouthWestFrench].pop()
            SouthWestFrench_len = SouthWestFrench_len - 1

        self.ml_id[BuyBakWines.SouthWestFrench] = np.repeat(['id_00', 'id_01'], SouthWestFrench_len/2)
        self.ml_ds[BuyBakWines.SouthWestFrench] = pd.date_range('2000-01-01', periods=SouthWestFrench_len, freq='D')
        self.ml_y[BuyBakWines.SouthWestFrench] = y[BuyBakWines.SouthWestFrench] #df_ohlcv[BuyBakWines.SouthWestFrench]['ema']
        print(f'[SouthWestFrench] len {len(y[BuyBakWines.SouthWestFrench])}, {len(self.ml_id[BuyBakWines.SouthWestFrench])} , {len(self.ml_ds[BuyBakWines.SouthWestFrench])} , {len(self.ml_y[BuyBakWines.SouthWestFrench])} ')
        self.ml_series[BuyBakWines.SouthWestFrench] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.SouthWestFrench],
            'ds': self.ml_ds[BuyBakWines.SouthWestFrench],
            'y': self.ml_y[BuyBakWines.SouthWestFrench],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.SouthWestFrench]) * 0.8)
        print('------------ SouthWestFrench train_size --------')
        print(train_size)
        X_train[BuyBakWines.SouthWestFrench], y_train[BuyBakWines.SouthWestFrench] = X[BuyBakWines.SouthWestFrench][:train_size], y[BuyBakWines.SouthWestFrench][:train_size]

        #######
        # French
        #######
        lf[BuyBakWines.French] = df_ohlcv[BuyBakWines.French][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.French] = np.array(lf[BuyBakWines.French]).tolist()
        y[BuyBakWines.French] = np.array(df_ohlcv[BuyBakWines.French]['ema']).tolist()


        # let's drop the last
        French_len = len(y[BuyBakWines.French])
        if French_len % 2 == 1:
            y[BuyBakWines.French].pop()
            French_len = French_len - 1

        self.ml_id[BuyBakWines.French] = np.repeat(['id_00', 'id_01'], French_len/2)
        self.ml_ds[BuyBakWines.French] = pd.date_range('2000-01-01', periods=French_len, freq='D')
        self.ml_y[BuyBakWines.French] = y[BuyBakWines.French] #df_ohlcv[BuyBakWines.French]['ema']
        print(f'[French] len {len(y[BuyBakWines.French])}, {len(self.ml_id[BuyBakWines.French])} , {len(self.ml_ds[BuyBakWines.French])} , {len(self.ml_y[BuyBakWines.French])} ')
        self.ml_series[BuyBakWines.French] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.French],
            'ds': self.ml_ds[BuyBakWines.French],
            'y': self.ml_y[BuyBakWines.French],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.French]) * 0.8)
        print('------------ French train_size --------')
        print(train_size)
        X_train[BuyBakWines.French], y_train[BuyBakWines.French] = X[BuyBakWines.French][:train_size], y[BuyBakWines.French][:train_size]

        #######
        # Jura
        #######
        lf[BuyBakWines.Jura] = df_ohlcv[BuyBakWines.Jura][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Jura] = np.array(lf[BuyBakWines.Jura]).tolist()
        y[BuyBakWines.Jura] = np.array(df_ohlcv[BuyBakWines.Jura]['ema']).tolist()


        # let's drop the last
        Jura_len = len(y[BuyBakWines.Jura])
        if Jura_len % 2 == 1:
            y[BuyBakWines.Jura].pop()
            Jura_len = Jura_len - 1

        self.ml_id[BuyBakWines.Jura] = np.repeat(['id_00', 'id_01'], Jura_len/2)
        self.ml_ds[BuyBakWines.Jura] = pd.date_range('2000-01-01', periods=Jura_len, freq='D')
        self.ml_y[BuyBakWines.Jura] = y[BuyBakWines.Jura] #df_ohlcv[BuyBakWines.Jura]['ema']
        print(f'[Jura] len {len(y[BuyBakWines.Jura])}, {len(self.ml_id[BuyBakWines.Jura])} , {len(self.ml_ds[BuyBakWines.Jura])} , {len(self.ml_y[BuyBakWines.Jura])} ')
        self.ml_series[BuyBakWines.Jura] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Jura],
            'ds': self.ml_ds[BuyBakWines.Jura],
            'y': self.ml_y[BuyBakWines.Jura],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Jura]) * 0.8)
        print('------------ Jura train_size --------')
        print(train_size)
        X_train[BuyBakWines.Jura], y_train[BuyBakWines.Jura] = X[BuyBakWines.Jura][:train_size], y[BuyBakWines.Jura][:train_size]

        #######
        # Savoy
        #######
        lf[BuyBakWines.Savoy] = df_ohlcv[BuyBakWines.Savoy][['Open', 'High', 'Low', 'Close']]
        X[BuyBakWines.Savoy] = np.array(lf[BuyBakWines.Savoy]).tolist()
        y[BuyBakWines.Savoy] = np.array(df_ohlcv[BuyBakWines.Savoy]['ema']).tolist()


        # let's drop the last
        Savoy_len = len(y[BuyBakWines.Savoy])
        if Savoy_len % 2 == 1:
            y[BuyBakWines.Savoy].pop()
            Savoy_len = Savoy_len - 1

        self.ml_id[BuyBakWines.Savoy] = np.repeat(['id_00', 'id_01'], Savoy_len/2)
        self.ml_ds[BuyBakWines.Savoy] = pd.date_range('2000-01-01', periods=Savoy_len, freq='D')
        self.ml_y[BuyBakWines.Savoy] = y[BuyBakWines.Savoy] #df_ohlcv[BuyBakWines.Savoy]['ema']
        print(f'[Savoy] len {len(y[BuyBakWines.Savoy])}, {len(self.ml_id[BuyBakWines.Savoy])} , {len(self.ml_ds[BuyBakWines.Savoy])} , {len(self.ml_y[BuyBakWines.Savoy])} ')
        self.ml_series[BuyBakWines.Savoy] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakWines.Savoy],
            'ds': self.ml_ds[BuyBakWines.Savoy],
            'y': self.ml_y[BuyBakWines.Savoy],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakWines.Savoy]) * 0.8)
        print('------------ Savoy train_size --------')
        print(train_size)
        X_train[BuyBakWines.Savoy], y_train[BuyBakWines.Savoy] = X[BuyBakWines.Savoy][:train_size], y[BuyBakWines.Savoy][:train_size]

        #######################
        # common to all wines
        #######################
        ml_models = [
            lgb.LGBMRegressor(random_state=0, verbosity=-1),
            # Add other models here if needed, e.g., LinearRegression()
        ]

        ###################################################################
        # Instantiate Alsace MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Alsace] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Alsace].fit(self.ml_series[BuyBakWines.Alsace])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Alsace])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Alsace])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Alsace] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Alsace].fit(X_train[BuyBakWines.Alsace], y_train[BuyBakWines.Alsace])
        print(self.model[BuyBakWines.Alsace])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Bordeaux MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Bordeaux] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Bordeaux].fit(self.ml_series[BuyBakWines.Bordeaux])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Bordeaux])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Bordeaux])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Bordeaux] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Bordeaux].fit(X_train[BuyBakWines.Bordeaux], y_train[BuyBakWines.Bordeaux])
        print(self.model[BuyBakWines.Bordeaux])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Burgundy MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Burgundy] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Burgundy].fit(self.ml_series[BuyBakWines.Burgundy])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Burgundy])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Burgundy])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Burgundy] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Burgundy].fit(X_train[BuyBakWines.Burgundy], y_train[BuyBakWines.Burgundy])
        print(self.model[BuyBakWines.Burgundy])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Champagne MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Champagne] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Champagne].fit(self.ml_series[BuyBakWines.Champagne])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Champagne])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Champagne])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Champagne] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Champagne].fit(X_train[BuyBakWines.Champagne], y_train[BuyBakWines.Champagne])
        print(self.model[BuyBakWines.Champagne])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Corsican MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Corsican] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Corsican].fit(self.ml_series[BuyBakWines.Corsican])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Corsican])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Corsican])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Corsican] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Corsican].fit(X_train[BuyBakWines.Corsican], y_train[BuyBakWines.Corsican])
        print(self.model[BuyBakWines.Corsican])
        print('------ init model done --------')

        ###################################################################
        # Instantiate SouthWestFrench MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.SouthWestFrench] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.SouthWestFrench].fit(self.ml_series[BuyBakWines.SouthWestFrench])

        print('------------ SouthWestFrench X_train --------')
        print(X_train[BuyBakWines.SouthWestFrench])
        print('------------ SouthWestFrench y_train --------')
        print(y_train[BuyBakWines.SouthWestFrench])

        # Create and train the XGBoost model
        # self.model[BuyBakWines.SouthWestFrench] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        # self.model[BuyBakWines.SouthWestFrench].fit(X_train[BuyBakWines.SouthWestFrench], y_train[BuyBakWines.SouthWestFrench])
        # print(self.model[BuyBakWines.SouthWestFrench])
        # print('------ SouthWestFrench init model done --------')

        ###################################################################
        # Instantiate French MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.French] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.French].fit(self.ml_series[BuyBakWines.French])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.French])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.French])

        # Create and train the XGBoost model
        self.model[BuyBakWines.French] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.French].fit(X_train[BuyBakWines.French], y_train[BuyBakWines.French])
        print(self.model[BuyBakWines.French])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Jura MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Jura] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Jura].fit(self.ml_series[BuyBakWines.Jura])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Jura])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Jura])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Jura] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Jura].fit(X_train[BuyBakWines.Jura], y_train[BuyBakWines.Jura])
        print(self.model[BuyBakWines.Jura])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Savoy MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakWines.Savoy] = MLForecast(
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
        self.ml_forecaster[BuyBakWines.Savoy].fit(self.ml_series[BuyBakWines.Savoy])

        # print('------------ X_train --------')
        # print(X_train[BuyBakWines.Savoy])
        # print('------------ y_train --------')
        # print(y_train[BuyBakWines.Savoy])

        # Create and train the XGBoost model
        self.model[BuyBakWines.Savoy] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakWines.Savoy].fit(X_train[BuyBakWines.Savoy], y_train[BuyBakWines.Savoy])
        print(self.model[BuyBakWines.Savoy])
        print('------ init model done --------')


    def extract_bbk_time_series_from_prompt(self, input: str) -> []:
        """Extract the BuyBakTimeSeries structure from the input """
        print("Inside extract_bbk_time_series_from_prompt")
        print(input)
        llm = OpenAI("gpt-4.1")
        sllm = llm.as_structured_llm(BuyBakTimeSeries)
        response = sllm.complete(input)
        print(response)
        return response.text


    def get_mean_squared_error(self) -> float:
        """Get the MSE from the live prediction ."""
        return self.mse[BuyBakWines.Alsace]

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
            y_pred_live = self.model[BuyBakWines.Alsace].predict(X_live)
            print('---------- y_pred_live  -------------')
            print(y_pred_live)
            self.mse[BuyBakWines.Alsace] = mean_squared_error(y_live, y_pred_live)
            print(f'---------- self.mse  {self.mse[BuyBakWines.Alsace]}')
            return y_pred_live
        except:
            return "Exception thrown parsing argin JSON"

    def buybak_model_forecast(self, index: int, argin: int) -> []:
        """MLForecaster: Use the ml_forecaster model. 'index' defines which ml_forecaster to use based on the BuyBakWines IntEnum range(0-8). Predict the next N values, based on the 'argin' input, and return."""

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

    def get_buybak_wines_enum(self) -> str:
        return "{'WinesEnum': [ {'Alsace': 0}, {'Bordeaux': 1}, {'Burgundy': 2}, {'Champagne':  3}, {'Corsican': 4}, {'French': 5}, {'SouthWestFrench': 6}, {'Jura': 7}, {'Savoy': 8}]}"
