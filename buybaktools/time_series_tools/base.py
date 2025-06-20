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

class BuyBakFrenchWines(IntEnum):
    Alsace              = 0
    Bordeaux            = 1
    Burgundy            = 2
    Champagne           = 3
    Corsican            = 4
    French              = 5
    SouthWestFrench     = 6
    Jura                = 7
    Savoy               = 8
    Ahr                 = 9
    Baden               = 10
    Franconia           = 11
    Hessische           = 12
    Kaiserstuhl         = 13
    Mosel               = 14
    Rheinhessen         = 15

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
    model           = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]

    # MLForecast Regressor Model
    ml_id           = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
    ml_ds           = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
    ml_y            = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
    ml_series       = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
    ml_predict      = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
    ml_forecaster   = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]



    def __init__(self):
        print('BBK: Initializing Time Series')

        #####################
        # reverse_Alsace.cv
        #####################
        df_ohlcv = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
        X = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
        y = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
        lf = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
        X_train = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]
        y_train = [ BuyBakFrenchWines.Alsace, BuyBakFrenchWines.Bordeaux, BuyBakFrenchWines.Burgundy, BuyBakFrenchWines.Champagne, BuyBakFrenchWines.Corsican, BuyBakFrenchWines.SouthWestFrench, BuyBakFrenchWines.French, BuyBakFrenchWines.Jura, BuyBakFrenchWines.Savoy, BuyBakFrenchWines.Ahr ]

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Alsace] = pd.read_csv("./TradesCSV/reverse_Alsace.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Alsace]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Alsace]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Alsace]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Alsace]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Alsace]['sma'] = df_ohlcv[BuyBakFrenchWines.Alsace]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Alsace]['ema'] = df_ohlcv[BuyBakFrenchWines.Alsace]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Alsace]['volatility'] = df_ohlcv[BuyBakFrenchWines.Alsace]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Alsace].dropna(inplace=True)

        #####################
        # reverse_Bordeaux.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Bordeaux] = pd.read_csv("./TradesCSV/reverse_Bordeaux.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Bordeaux]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Bordeaux]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Bordeaux]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Bordeaux]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Bordeaux]['sma'] = df_ohlcv[BuyBakFrenchWines.Bordeaux]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Bordeaux]['ema'] = df_ohlcv[BuyBakFrenchWines.Bordeaux]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Bordeaux]['volatility'] = df_ohlcv[BuyBakFrenchWines.Bordeaux]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Bordeaux].dropna(inplace=True)

        #####################
        # reverse_Burgundy.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Burgundy] = pd.read_csv("./TradesCSV/reverse_Burgundy.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Burgundy]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Burgundy]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Burgundy]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Burgundy]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Burgundy]['sma'] = df_ohlcv[BuyBakFrenchWines.Burgundy]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Burgundy]['ema'] = df_ohlcv[BuyBakFrenchWines.Burgundy]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Burgundy]['volatility'] = df_ohlcv[BuyBakFrenchWines.Burgundy]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Burgundy].dropna(inplace=True)

        #####################
        # reverse_Champagne.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Champagne] = pd.read_csv("./TradesCSV/reverse_Champagne.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Champagne]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Champagne]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Champagne]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Champagne]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Champagne]['sma'] = df_ohlcv[BuyBakFrenchWines.Champagne]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Champagne]['ema'] = df_ohlcv[BuyBakFrenchWines.Champagne]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Champagne]['volatility'] = df_ohlcv[BuyBakFrenchWines.Champagne]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Champagne].dropna(inplace=True)

        #####################
        # reverse_Corsican.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Corsican] = pd.read_csv("./TradesCSV/reverse_Corsican.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Corsican]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Corsican]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Corsican]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Corsican]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Corsican]['sma'] = df_ohlcv[BuyBakFrenchWines.Corsican]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Corsican]['ema'] = df_ohlcv[BuyBakFrenchWines.Corsican]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Corsican]['volatility'] = df_ohlcv[BuyBakFrenchWines.Corsican]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Corsican].dropna(inplace=True)

        #####################
        # reverse_SouthWestFrench.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench] = pd.read_csv("./TradesCSV/reverse_SouthWestFrench.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['sma'] = df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['ema'] = df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['volatility'] = df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.SouthWestFrench].dropna(inplace=True)

        #####################
        # reverse_French.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.French] = pd.read_csv("./TradesCSV/reverse_French.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.French]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.French]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.French]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.French]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.French]['sma'] = df_ohlcv[BuyBakFrenchWines.French]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.French]['ema'] = df_ohlcv[BuyBakFrenchWines.French]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.French]['volatility'] = df_ohlcv[BuyBakFrenchWines.French]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.French].dropna(inplace=True)

        #####################
        # reverse_Jura.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Jura] = pd.read_csv("./TradesCSV/reverse_Jura.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Jura]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Jura]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Jura]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Jura]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Jura]['sma'] = df_ohlcv[BuyBakFrenchWines.Jura]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Jura]['ema'] = df_ohlcv[BuyBakFrenchWines.Jura]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Jura]['volatility'] = df_ohlcv[BuyBakFrenchWines.Jura]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Jura].dropna(inplace=True)

        #####################
        # reverse_Savoy.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Savoy] = pd.read_csv("./TradesCSV/reverse_Savoy.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Savoy]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Savoy]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Savoy]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Savoy]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Savoy]['sma'] = df_ohlcv[BuyBakFrenchWines.Savoy]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Savoy]['ema'] = df_ohlcv[BuyBakFrenchWines.Savoy]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Savoy]['volatility'] = df_ohlcv[BuyBakFrenchWines.Savoy]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Savoy].dropna(inplace=True)

        #####################
        # reverse_Ahr.cv
        #####################

        # 1. Load and clean data
        df_ohlcv[BuyBakFrenchWines.Ahr] = pd.read_csv("./TradesCSV/reverse_Ahr.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv[BuyBakFrenchWines.Ahr]['rsi'] = ta.momentum.RSIIndicator(df_ohlcv[BuyBakFrenchWines.Ahr]['Close']).rsi()
        df_ohlcv[BuyBakFrenchWines.Ahr]['macd'] = ta.trend.MACD(df_ohlcv[BuyBakFrenchWines.Ahr]['Close']).macd()
        df_ohlcv[BuyBakFrenchWines.Ahr]['sma'] = df_ohlcv[BuyBakFrenchWines.Ahr]['Close'].rolling(14).mean()
        df_ohlcv[BuyBakFrenchWines.Ahr]['ema'] = df_ohlcv[BuyBakFrenchWines.Ahr]['Close'].ewm(span=14).mean()
        df_ohlcv[BuyBakFrenchWines.Ahr]['volatility'] = df_ohlcv[BuyBakFrenchWines.Ahr]['Close'].rolling(14).std()
        df_ohlcv[BuyBakFrenchWines.Ahr].dropna(inplace=True)

        ##############################
        # prepare X y
        ##############################

        #######
        # Alsace
        #######
        lf[BuyBakFrenchWines.Alsace] = df_ohlcv[BuyBakFrenchWines.Alsace][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Alsace] = np.array(lf[BuyBakFrenchWines.Alsace]).tolist()
        y[BuyBakFrenchWines.Alsace] = np.array(df_ohlcv[BuyBakFrenchWines.Alsace]['ema']).tolist()

        # let's drop the last
        Alsace_len = len(y[BuyBakFrenchWines.Alsace])
        if Alsace_len % 2 == 1:
            y[BuyBakFrenchWines.Alsace].pop()
            Alsace_len = Alsace_len - 1

        self.ml_id[BuyBakFrenchWines.Alsace] = np.repeat(['id_00', 'id_01'], Alsace_len/2)
        self.ml_ds[BuyBakFrenchWines.Alsace] = pd.date_range('2000-01-01', periods=Alsace_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Alsace] = df_ohlcv[BuyBakFrenchWines.Alsace]['ema']
        print(f'[Alsace] len {len(y[BuyBakFrenchWines.Alsace])}, {len(self.ml_id[BuyBakFrenchWines.Alsace])} , {len(self.ml_ds[BuyBakFrenchWines.Alsace])} , {len(self.ml_y[BuyBakFrenchWines.Alsace])} ')
        self.ml_series[BuyBakFrenchWines.Alsace] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Alsace],
            'ds': self.ml_ds[BuyBakFrenchWines.Alsace],
            'y': self.ml_y[BuyBakFrenchWines.Alsace],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Alsace]) * 0.8)
        print('------------ Alsace train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Alsace], y_train[BuyBakFrenchWines.Alsace] = X[BuyBakFrenchWines.Alsace][:train_size], y[BuyBakFrenchWines.Alsace][:train_size]

        #######
        # Bordeaux
        #######
        lf[BuyBakFrenchWines.Bordeaux] = df_ohlcv[BuyBakFrenchWines.Bordeaux][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Bordeaux] = np.array(lf[BuyBakFrenchWines.Bordeaux]).tolist()
        y[BuyBakFrenchWines.Bordeaux] = np.array(df_ohlcv[BuyBakFrenchWines.Bordeaux]['ema']).tolist()

        # let's drop the last
        Bordeaux_len = len(y[BuyBakFrenchWines.Bordeaux])
        if Bordeaux_len % 2 == 1:
            y[BuyBakFrenchWines.Bordeaux].pop()
            Bordeaux_len = Bordeaux_len - 1

        self.ml_id[BuyBakFrenchWines.Bordeaux] = np.repeat(['id_00', 'id_01'], Bordeaux_len/2)
        self.ml_ds[BuyBakFrenchWines.Bordeaux] = pd.date_range('2000-01-01', periods=Bordeaux_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Bordeaux] = y[BuyBakFrenchWines.Bordeaux] #df_ohlcv[BuyBakFrenchWines.Bordeaux]['ema']

        print(f'[Bordeaux] len {len(y[BuyBakFrenchWines.Bordeaux])}, {len(self.ml_id[BuyBakFrenchWines.Bordeaux])} , {len(self.ml_ds[BuyBakFrenchWines.Bordeaux])} , {len(self.ml_y[BuyBakFrenchWines.Bordeaux])} ')
        self.ml_series[BuyBakFrenchWines.Bordeaux] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Bordeaux],
            'ds': self.ml_ds[BuyBakFrenchWines.Bordeaux],
            'y': self.ml_y[BuyBakFrenchWines.Bordeaux],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Bordeaux]) * 0.8)
        print('------------ Bordeaux train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Bordeaux], y_train[BuyBakFrenchWines.Bordeaux] = X[BuyBakFrenchWines.Bordeaux][:train_size], y[BuyBakFrenchWines.Bordeaux][:train_size]

        #######
        # Burgundy
        #######
        lf[BuyBakFrenchWines.Burgundy] = df_ohlcv[BuyBakFrenchWines.Burgundy][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Burgundy] = np.array(lf[BuyBakFrenchWines.Burgundy]).tolist()
        y[BuyBakFrenchWines.Burgundy] = np.array(df_ohlcv[BuyBakFrenchWines.Burgundy]['ema']).tolist()


        # let's drop the last
        Burgundy_len = len(y[BuyBakFrenchWines.Burgundy])
        if Burgundy_len % 2 == 1:
            y[BuyBakFrenchWines.Burgundy].pop()
            Burgundy_len = Burgundy_len - 1

        self.ml_id[BuyBakFrenchWines.Burgundy] = np.repeat(['id_00', 'id_01'], Burgundy_len/2)
        self.ml_ds[BuyBakFrenchWines.Burgundy] = pd.date_range('2000-01-01', periods=Burgundy_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Burgundy] = y[BuyBakFrenchWines.Burgundy] #df_ohlcv[BuyBakFrenchWines.Burgundy]['ema']
        print(f'[Burgundy] len {len(y[BuyBakFrenchWines.Burgundy])}, {len(self.ml_id[BuyBakFrenchWines.Burgundy])} , {len(self.ml_ds[BuyBakFrenchWines.Burgundy])} , {len(self.ml_y[BuyBakFrenchWines.Burgundy])} ')
        self.ml_series[BuyBakFrenchWines.Burgundy] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Burgundy],
            'ds': self.ml_ds[BuyBakFrenchWines.Burgundy],
            'y': self.ml_y[BuyBakFrenchWines.Burgundy],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Burgundy]) * 0.8)
        print('------------ Burgundy train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Burgundy], y_train[BuyBakFrenchWines.Burgundy] = X[BuyBakFrenchWines.Burgundy][:train_size], y[BuyBakFrenchWines.Burgundy][:train_size]

        #######
        # Champagne
        #######
        lf[BuyBakFrenchWines.Champagne] = df_ohlcv[BuyBakFrenchWines.Champagne][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Champagne] = np.array(lf[BuyBakFrenchWines.Champagne]).tolist()
        y[BuyBakFrenchWines.Champagne] = np.array(df_ohlcv[BuyBakFrenchWines.Champagne]['ema']).tolist()


        # let's drop the last
        Champagne_len = len(y[BuyBakFrenchWines.Champagne])
        if Champagne_len % 2 == 1:
            y[BuyBakFrenchWines.Champagne].pop()
            Champagne_len = Champagne_len - 1

        self.ml_id[BuyBakFrenchWines.Champagne] = np.repeat(['id_00', 'id_01'], Champagne_len/2)
        self.ml_ds[BuyBakFrenchWines.Champagne] = pd.date_range('2000-01-01', periods=Champagne_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Champagne] = y[BuyBakFrenchWines.Champagne] #df_ohlcv[BuyBakFrenchWines.Champagne]['ema']
        print(f'[Champagne] len {len(y[BuyBakFrenchWines.Champagne])}, {len(self.ml_id[BuyBakFrenchWines.Champagne])} , {len(self.ml_ds[BuyBakFrenchWines.Champagne])} , {len(self.ml_y[BuyBakFrenchWines.Champagne])} ')
        self.ml_series[BuyBakFrenchWines.Champagne] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Champagne],
            'ds': self.ml_ds[BuyBakFrenchWines.Champagne],
            'y': self.ml_y[BuyBakFrenchWines.Champagne],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Champagne]) * 0.8)
        print('------------ Champagne train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Champagne], y_train[BuyBakFrenchWines.Champagne] = X[BuyBakFrenchWines.Champagne][:train_size], y[BuyBakFrenchWines.Champagne][:train_size]

        #######
        # Corsican
        #######
        lf[BuyBakFrenchWines.Corsican] = df_ohlcv[BuyBakFrenchWines.Corsican][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Corsican] = np.array(lf[BuyBakFrenchWines.Corsican]).tolist()
        y[BuyBakFrenchWines.Corsican] = np.array(df_ohlcv[BuyBakFrenchWines.Corsican]['ema']).tolist()


        # let's drop the last
        Corsican_len = len(y[BuyBakFrenchWines.Corsican])
        if Corsican_len % 2 == 1:
            y[BuyBakFrenchWines.Corsican].pop()
            Corsican_len = Corsican_len - 1

        self.ml_id[BuyBakFrenchWines.Corsican] = np.repeat(['id_00', 'id_01'], Corsican_len/2)
        self.ml_ds[BuyBakFrenchWines.Corsican] = pd.date_range('2000-01-01', periods=Corsican_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Corsican] = y[BuyBakFrenchWines.Corsican] #df_ohlcv[BuyBakFrenchWines.Corsican]['ema']
        print(f'[Corsican] len {len(y[BuyBakFrenchWines.Corsican])}, {len(self.ml_id[BuyBakFrenchWines.Corsican])} , {len(self.ml_ds[BuyBakFrenchWines.Corsican])} , {len(self.ml_y[BuyBakFrenchWines.Corsican])} ')
        self.ml_series[BuyBakFrenchWines.Corsican] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Corsican],
            'ds': self.ml_ds[BuyBakFrenchWines.Corsican],
            'y': self.ml_y[BuyBakFrenchWines.Corsican],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Corsican]) * 0.8)
        print('------------ Corsican train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Corsican], y_train[BuyBakFrenchWines.Corsican] = X[BuyBakFrenchWines.Corsican][:train_size], y[BuyBakFrenchWines.Corsican][:train_size]

        #######
        # SouthWestFrench
        #######
        lf[BuyBakFrenchWines.SouthWestFrench] = df_ohlcv[BuyBakFrenchWines.SouthWestFrench][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.SouthWestFrench] = np.array(lf[BuyBakFrenchWines.SouthWestFrench]).tolist()
        y[BuyBakFrenchWines.SouthWestFrench] = np.array(df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['ema']).tolist()


        # let's drop the last
        SouthWestFrench_len = len(y[BuyBakFrenchWines.SouthWestFrench])
        if SouthWestFrench_len % 2 == 1:
            y[BuyBakFrenchWines.SouthWestFrench].pop()
            SouthWestFrench_len = SouthWestFrench_len - 1

        self.ml_id[BuyBakFrenchWines.SouthWestFrench] = np.repeat(['id_00', 'id_01'], SouthWestFrench_len/2)
        self.ml_ds[BuyBakFrenchWines.SouthWestFrench] = pd.date_range('2000-01-01', periods=SouthWestFrench_len, freq='D')
        self.ml_y[BuyBakFrenchWines.SouthWestFrench] = y[BuyBakFrenchWines.SouthWestFrench] #df_ohlcv[BuyBakFrenchWines.SouthWestFrench]['ema']
        print(f'[SouthWestFrench] len {len(y[BuyBakFrenchWines.SouthWestFrench])}, {len(self.ml_id[BuyBakFrenchWines.SouthWestFrench])} , {len(self.ml_ds[BuyBakFrenchWines.SouthWestFrench])} , {len(self.ml_y[BuyBakFrenchWines.SouthWestFrench])} ')
        self.ml_series[BuyBakFrenchWines.SouthWestFrench] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.SouthWestFrench],
            'ds': self.ml_ds[BuyBakFrenchWines.SouthWestFrench],
            'y': self.ml_y[BuyBakFrenchWines.SouthWestFrench],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.SouthWestFrench]) * 0.8)
        print('------------ SouthWestFrench train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.SouthWestFrench], y_train[BuyBakFrenchWines.SouthWestFrench] = X[BuyBakFrenchWines.SouthWestFrench][:train_size], y[BuyBakFrenchWines.SouthWestFrench][:train_size]

        #######
        # French
        #######
        lf[BuyBakFrenchWines.French] = df_ohlcv[BuyBakFrenchWines.French][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.French] = np.array(lf[BuyBakFrenchWines.French]).tolist()
        y[BuyBakFrenchWines.French] = np.array(df_ohlcv[BuyBakFrenchWines.French]['ema']).tolist()


        # let's drop the last
        French_len = len(y[BuyBakFrenchWines.French])
        if French_len % 2 == 1:
            y[BuyBakFrenchWines.French].pop()
            French_len = French_len - 1

        self.ml_id[BuyBakFrenchWines.French] = np.repeat(['id_00', 'id_01'], French_len/2)
        self.ml_ds[BuyBakFrenchWines.French] = pd.date_range('2000-01-01', periods=French_len, freq='D')
        self.ml_y[BuyBakFrenchWines.French] = y[BuyBakFrenchWines.French] #df_ohlcv[BuyBakFrenchWines.French]['ema']
        print(f'[French] len {len(y[BuyBakFrenchWines.French])}, {len(self.ml_id[BuyBakFrenchWines.French])} , {len(self.ml_ds[BuyBakFrenchWines.French])} , {len(self.ml_y[BuyBakFrenchWines.French])} ')
        self.ml_series[BuyBakFrenchWines.French] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.French],
            'ds': self.ml_ds[BuyBakFrenchWines.French],
            'y': self.ml_y[BuyBakFrenchWines.French],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.French]) * 0.8)
        print('------------ French train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.French], y_train[BuyBakFrenchWines.French] = X[BuyBakFrenchWines.French][:train_size], y[BuyBakFrenchWines.French][:train_size]

        #######
        # Jura
        #######
        lf[BuyBakFrenchWines.Jura] = df_ohlcv[BuyBakFrenchWines.Jura][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Jura] = np.array(lf[BuyBakFrenchWines.Jura]).tolist()
        y[BuyBakFrenchWines.Jura] = np.array(df_ohlcv[BuyBakFrenchWines.Jura]['ema']).tolist()


        # let's drop the last
        Jura_len = len(y[BuyBakFrenchWines.Jura])
        if Jura_len % 2 == 1:
            y[BuyBakFrenchWines.Jura].pop()
            Jura_len = Jura_len - 1

        self.ml_id[BuyBakFrenchWines.Jura] = np.repeat(['id_00', 'id_01'], Jura_len/2)
        self.ml_ds[BuyBakFrenchWines.Jura] = pd.date_range('2000-01-01', periods=Jura_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Jura] = y[BuyBakFrenchWines.Jura] #df_ohlcv[BuyBakFrenchWines.Jura]['ema']
        print(f'[Jura] len {len(y[BuyBakFrenchWines.Jura])}, {len(self.ml_id[BuyBakFrenchWines.Jura])} , {len(self.ml_ds[BuyBakFrenchWines.Jura])} , {len(self.ml_y[BuyBakFrenchWines.Jura])} ')
        self.ml_series[BuyBakFrenchWines.Jura] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Jura],
            'ds': self.ml_ds[BuyBakFrenchWines.Jura],
            'y': self.ml_y[BuyBakFrenchWines.Jura],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Jura]) * 0.8)
        print('------------ Jura train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Jura], y_train[BuyBakFrenchWines.Jura] = X[BuyBakFrenchWines.Jura][:train_size], y[BuyBakFrenchWines.Jura][:train_size]

        #######
        # Savoy
        #######
        lf[BuyBakFrenchWines.Savoy] = df_ohlcv[BuyBakFrenchWines.Savoy][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Savoy] = np.array(lf[BuyBakFrenchWines.Savoy]).tolist()
        y[BuyBakFrenchWines.Savoy] = np.array(df_ohlcv[BuyBakFrenchWines.Savoy]['ema']).tolist()


        # let's drop the last
        Savoy_len = len(y[BuyBakFrenchWines.Savoy])
        if Savoy_len % 2 == 1:
            y[BuyBakFrenchWines.Savoy].pop()
            Savoy_len = Savoy_len - 1

        self.ml_id[BuyBakFrenchWines.Savoy] = np.repeat(['id_00', 'id_01'], Savoy_len/2)
        self.ml_ds[BuyBakFrenchWines.Savoy] = pd.date_range('2000-01-01', periods=Savoy_len, freq='D')
        self.ml_y[BuyBakFrenchWines.Savoy] = y[BuyBakFrenchWines.Savoy] #df_ohlcv[BuyBakFrenchWines.Savoy]['ema']
        print(f'[Savoy] len {len(y[BuyBakFrenchWines.Savoy])}, {len(self.ml_id[BuyBakFrenchWines.Savoy])} , {len(self.ml_ds[BuyBakFrenchWines.Savoy])} , {len(self.ml_y[BuyBakFrenchWines.Savoy])} ')
        self.ml_series[BuyBakFrenchWines.Savoy] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Savoy],
            'ds': self.ml_ds[BuyBakFrenchWines.Savoy],
            'y': self.ml_y[BuyBakFrenchWines.Savoy],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Savoy]) * 0.8)
        print('------------ Savoy train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Savoy], y_train[BuyBakFrenchWines.Savoy] = X[BuyBakFrenchWines.Savoy][:train_size], y[BuyBakFrenchWines.Savoy][:train_size]

        #######
        # Ahr
        #######
        lf[BuyBakFrenchWines.Ahr] = df_ohlcv[BuyBakFrenchWines.Ahr][['Open', 'High', 'Low', 'Close']]
        X[BuyBakFrenchWines.Ahr] = np.array(lf[BuyBakFrenchWines.Ahr]).tolist()
        y[BuyBakFrenchWines.Ahr] = np.array(df_ohlcv[BuyBakFrenchWines.Ahr]['ema']).tolist()


        # let's drop the last
        Ahr = len(y[BuyBakFrenchWines.Ahr])
        if Ahr % 2 == 1:
            y[BuyBakFrenchWines.Ahr].pop()
            Ahr = Ahr - 1

        self.ml_id[BuyBakFrenchWines.Ahr] = np.repeat(['id_00', 'id_01'], Ahr/2)
        self.ml_ds[BuyBakFrenchWines.Ahr] = pd.date_range('2000-01-01', periods=Ahr, freq='D')
        self.ml_y[BuyBakFrenchWines.Ahr] = y[BuyBakFrenchWines.Ahr] #df_ohlcv[BuyBakFrenchWines.Ahr]['ema']
        print(f'[Ahr] len {len(y[BuyBakFrenchWines.Ahr])}, {len(self.ml_id[BuyBakFrenchWines.Ahr])} , {len(self.ml_ds[BuyBakFrenchWines.Ahr])} , {len(self.ml_y[BuyBakFrenchWines.Ahr])} ')
        self.ml_series[BuyBakFrenchWines.Ahr] = pd.DataFrame({
            'unique_id': self.ml_id[BuyBakFrenchWines.Ahr],
            'ds': self.ml_ds[BuyBakFrenchWines.Ahr],
            'y': self.ml_y[BuyBakFrenchWines.Ahr],
        })

        # Split the data into training and testing sets
        train_size = int(len(X[BuyBakFrenchWines.Ahr]) * 0.8)
        print('------------ Ahr train_size --------')
        print(train_size)
        X_train[BuyBakFrenchWines.Ahr], y_train[BuyBakFrenchWines.Ahr] = X[BuyBakFrenchWines.Ahr][:train_size], y[BuyBakFrenchWines.Ahr][:train_size]

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
        self.ml_forecaster[BuyBakFrenchWines.Alsace] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Alsace].fit(self.ml_series[BuyBakFrenchWines.Alsace])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Alsace])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Alsace])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Alsace] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Alsace].fit(X_train[BuyBakFrenchWines.Alsace], y_train[BuyBakFrenchWines.Alsace])
        print(self.model[BuyBakFrenchWines.Alsace])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Bordeaux MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Bordeaux] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Bordeaux].fit(self.ml_series[BuyBakFrenchWines.Bordeaux])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Bordeaux])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Bordeaux])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Bordeaux] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Bordeaux].fit(X_train[BuyBakFrenchWines.Bordeaux], y_train[BuyBakFrenchWines.Bordeaux])
        print(self.model[BuyBakFrenchWines.Bordeaux])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Burgundy MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Burgundy] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Burgundy].fit(self.ml_series[BuyBakFrenchWines.Burgundy])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Burgundy])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Burgundy])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Burgundy] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Burgundy].fit(X_train[BuyBakFrenchWines.Burgundy], y_train[BuyBakFrenchWines.Burgundy])
        print(self.model[BuyBakFrenchWines.Burgundy])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Champagne MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Champagne] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Champagne].fit(self.ml_series[BuyBakFrenchWines.Champagne])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Champagne])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Champagne])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Champagne] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Champagne].fit(X_train[BuyBakFrenchWines.Champagne], y_train[BuyBakFrenchWines.Champagne])
        print(self.model[BuyBakFrenchWines.Champagne])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Corsican MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Corsican] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Corsican].fit(self.ml_series[BuyBakFrenchWines.Corsican])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Corsican])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Corsican])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Corsican] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Corsican].fit(X_train[BuyBakFrenchWines.Corsican], y_train[BuyBakFrenchWines.Corsican])
        print(self.model[BuyBakFrenchWines.Corsican])
        print('------ init model done --------')

        ###################################################################
        # Instantiate SouthWestFrench MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.SouthWestFrench] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.SouthWestFrench].fit(self.ml_series[BuyBakFrenchWines.SouthWestFrench])

        print('------------ SouthWestFrench X_train --------')
        print(X_train[BuyBakFrenchWines.SouthWestFrench])
        print('------------ SouthWestFrench y_train --------')
        print(y_train[BuyBakFrenchWines.SouthWestFrench])

        # Create and train the XGBoost model
        # self.model[BuyBakFrenchWines.SouthWestFrench] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        # self.model[BuyBakFrenchWines.SouthWestFrench].fit(X_train[BuyBakFrenchWines.SouthWestFrench], y_train[BuyBakFrenchWines.SouthWestFrench])
        # print(self.model[BuyBakFrenchWines.SouthWestFrench])
        # print('------ SouthWestFrench init model done --------')

        ###################################################################
        # Instantiate French MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.French] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.French].fit(self.ml_series[BuyBakFrenchWines.French])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.French])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.French])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.French] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.French].fit(X_train[BuyBakFrenchWines.French], y_train[BuyBakFrenchWines.French])
        print(self.model[BuyBakFrenchWines.French])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Jura MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Jura] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Jura].fit(self.ml_series[BuyBakFrenchWines.Jura])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Jura])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Jura])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Jura] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Jura].fit(X_train[BuyBakFrenchWines.Jura], y_train[BuyBakFrenchWines.Jura])
        print(self.model[BuyBakFrenchWines.Jura])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Savoy MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Savoy] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Savoy].fit(self.ml_series[BuyBakFrenchWines.Savoy])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Savoy])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Savoy])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Savoy] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Savoy].fit(X_train[BuyBakFrenchWines.Savoy], y_train[BuyBakFrenchWines.Savoy])
        print(self.model[BuyBakFrenchWines.Savoy])
        print('------ init model done --------')

        ###################################################################
        # Instantiate Ahr MLForecast object
        ###################################################################
        self.ml_forecaster[BuyBakFrenchWines.Ahr] = MLForecast(
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
        self.ml_forecaster[BuyBakFrenchWines.Ahr].fit(self.ml_series[BuyBakFrenchWines.Ahr])

        # print('------------ X_train --------')
        # print(X_train[BuyBakFrenchWines.Ahr])
        # print('------------ y_train --------')
        # print(y_train[BuyBakFrenchWines.Ahr])

        # Create and train the XGBoost model
        self.model[BuyBakFrenchWines.Ahr] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model[BuyBakFrenchWines.Ahr].fit(X_train[BuyBakFrenchWines.Ahr], y_train[BuyBakFrenchWines.Ahr])
        print(self.model[BuyBakFrenchWines.Ahr])
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
        return self.mse[BuyBakFrenchWines.Alsace]

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
            y_pred_live = self.model[BuyBakFrenchWines.Alsace].predict(X_live)
            print('---------- y_pred_live  -------------')
            print(y_pred_live)
            self.mse[BuyBakFrenchWines.Alsace] = mean_squared_error(y_live, y_pred_live)
            print(f'---------- self.mse  {self.mse[BuyBakFrenchWines.Alsace]}')
            return y_pred_live
        except:
            return "Exception thrown parsing argin JSON"

    def buybak_model_forecast(self, index: int, argin: int) -> []:
        """MLForecaster: Use the ml_forecaster model. 'index' defines which ml_forecaster to use based on the BuyBakFrenchWines IntEnum range(0-8). Predict the next N values, based on the 'argin' input, and return."""

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
        return "{'WinesEnum': [ {'Alsace': 0}, {'Bordeaux': 1}, {'Burgundy': 2}, {'Champagne':  3}, {'Corsican': 4}, {'French': 5}, {'SouthWestFrench': 6}, {'Jura': 7}, {'Savoy': 8}, {'Ahr': 9}]}"
