"""BuyBak Appointment Booking Tool Spec."""

import numpy as np
import matplotlib.pyplot as plt
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

from llama_index.readers.web import AgentQLWebReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)


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

class BuyBakTimeSeriesToolSpec(BaseToolSpec):
    """BuyBak Apt Booking Tool spec."""

    spec_functions = [
        "extract_bbk_time_series_from_prompt",
        "get_bbk_ohlcv", 
        "get_bbk_indicators",
        "get_mean_squared_error",
        "buybak_model_predict",
        "buybak_model_forecast"
    ]
    query_engine                = None
    df_ohlcv                    = {}
    X                           = {}
    y                           = {}
    X_train                     = {}
    y_train                     = {}
    X_test                      = {}
    y_test                      = {}
    train_size                  = 0
    mse                         = 0.0
    storage_dir                 = "./storage-buybak-time-series"
    # MLForecast variables
    ml_id                       = {}
    ml_ds                       = {}
    ml_y                        = {}
    ml_series                   = {}
    ml_predict                    = {}
    ml_models                   = []
    ml_forecaster                = None



    def __init__(self):
        print('BBK: Initializing Time Series')

        # 1. Load and clean data
        df_ohlcv = pd.read_csv("./TradesCSV/reverse_goog.csv")
        # df = pd.read_csv("TradesCSV/small_goog.csv")
        print('df')

        # 2. Feature Engineering
        df_ohlcv['rsi'] = ta.momentum.RSIIndicator(df_ohlcv['Close']).rsi()
        df_ohlcv['macd'] = ta.trend.MACD(df_ohlcv['Close']).macd()
        df_ohlcv['sma'] = df_ohlcv['Close'].rolling(14).mean()
        df_ohlcv['ema'] = df_ohlcv['Close'].ewm(span=14).mean()
        df_ohlcv['volatility'] = df_ohlcv['Close'].rolling(14).std()
        df_ohlcv.dropna(inplace=True)

        lf = df_ohlcv[['Open', 'High', 'Low', 'Close']]
        self.X = np.array(lf).tolist()
        self.y = np.array(df_ohlcv['ema']).tolist()

        self.ml_id = np.repeat(['id_00', 'id_01'], 105)
        self.ml_ds = pd.date_range('2000-01-01', periods=210, freq='D')
        self.ml_y = df_ohlcv["ema"]
        self.ml_series = pd.DataFrame({
            'unique_id': self.ml_id,
            'ds': self.ml_ds,
            'y': self.ml_y,
        })
        self.ml_models = [
            lgb.LGBMRegressor(random_state=0, verbosity=-1),
            # Add other models here if needed, e.g., LinearRegression()
        ]

        # Split the data into training and testing sets
        self.train_size = int(len(self.X) * 0.8)
        self.X_train, self.y_train = self.X[:self.train_size], self.y[:self.train_size]
        self.X_test, self.y_test = self.X[self.train_size:], self.y[self.train_size:]


        # Instantiate MLForecast object
        self.ml_forecaster = MLForecast(
            models=self.ml_models,
            freq='D',
            lags=[7, 14],
            lag_transforms={
                1: [ExpandingMean()],
                7: [RollingMean(window_size=28)]
            },
            date_features=['dayofweek'],
            target_transforms=[Differences([1])]
        )
        self.ml_forecaster.fit(self.ml_series)

        print('------------ X_train --------')
        print(self.X_train)
        print('------------ y_train --------')
        print(self.y_train)
        print('------------ train_size --------')
        print(self.train_size)

        # Create and train the XGBoost model
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print(self.model)
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

    def get_bbk_ohlcv(self) -> []:
        """Get the OHLCV values from the dataframe ."""
        try:
            return self.df_ohlcv
        except:
            return f"OHLCV not found"

    def get_bbk_indicators(self) -> []:
        """Get the Indicators values from the dataframe ."""
        try:
            return self.df_ohlcv[['close', 'rsi', 'macd', 'sma', 'ema', 'volatility']]
        except:
            return f"OHLCV not found"

    def get_mean_squared_error(self) -> float:
        """Get the MSE from the live prediction ."""
        return self.mse

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
            y_pred_live = self.model.predict(X_live)
            print('---------- y_pred_live  -------------')
            print(y_pred_live)
            self.mse = mean_squared_error(y_live, y_pred_live)
            print(f'---------- self.mse  {self.mse}')
            return y_pred_live
        except:
            return "Exception thrown parsing argin JSON"

    def buybak_model_forecast(self) -> []:
        """MLForecaster: Use the ml_forecaster model to predict the next N values, and return."""

        try:
            self.ml_predict = self.ml_forecaster.predict(15)
            print('---------- ml_predict  -------------')
            print(self.ml_predict)
            filtered_df = self.ml_predict[(self.ml_predict['unique_id'] == "id_01")]
            filtered_df = filtered_df["LGBMRegressor"]
            return np.array(filtered_df).tolist()
        except:
            return "Exception thrown in ml_forecaster"

