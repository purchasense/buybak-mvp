"""BuyBak Appointment Booking Tool Spec."""

import numpy as np
import ta  # pip install ta
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

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

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool


class BuyBakFrenchWinesToolSpec(BaseToolSpec):
    """BuyBak French Wines Tool spec."""

    spec_functions = [
        "get_query_engine_tools_for_french_wines",
    ]

    def __init__(self):
        print('BBK: Initializing Query Engine Tools for French Wines')
        storage_context_1 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/AlsaceWine")
        storage_context_2 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/BordeauxWine")
        storage_context_3 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/BurgundyWine")
        storage_context_4 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/ChampagneWine")
        storage_context_5 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/CorsicanWine")
        storage_context_6 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/FrenchWine")
        storage_context_7 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/JuraWine")
        storage_context_8 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/RhoneWine")
        storage_context_9 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/SavoyWine")
        storage_context_0 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/SouthWestFrenchWine")

        self.index1 = load_index_from_storage(storage_context_1)
        self.index2 = load_index_from_storage(storage_context_2)
        self.index3 = load_index_from_storage(storage_context_3)
        self.index4 = load_index_from_storage(storage_context_4)
        self.index5 = load_index_from_storage(storage_context_5)
        self.index6 = load_index_from_storage(storage_context_6)
        self.index7 = load_index_from_storage(storage_context_7)
        self.index8 = load_index_from_storage(storage_context_8)
        self.index9 = load_index_from_storage(storage_context_9)
        self.index0 = load_index_from_storage(storage_context_0)

        self.index_loaded = True
        print('------ init french wines query_engine_index done --------')


    def get_query_engine_tools_for_french_wines(self) -> []:
        """Extract the BuyBakTimeSeries structure from the input """
        print("Inside get_query_engine_tools_for_french_wines")
        engine_AlsaceWine =             self.index1.as_query_engine(similarity_top_k=3)
        engine_BordeauxWine =           self.index2.as_query_engine(similarity_top_k=3)
        engine_BurgundyWine =           self.index3.as_query_engine(similarity_top_k=3)
        engine_ChampagneWine =          self.index4.as_query_engine(similarity_top_k=3)
        engine_CorsicanWine =           self.index5.as_query_engine(similarity_top_k=3)
        engine_FrenchWine =             self.index6.as_query_engine(similarity_top_k=3)
        engine_JuraWine =               self.index7.as_query_engine(similarity_top_k=3)
        engine_RhoneWine =              self.index8.as_query_engine(similarity_top_k=3)
        engine_SavoyWine =              self.index9.as_query_engine(similarity_top_k=3)
        engine_SouthWestFrenchWine =    self.index0.as_query_engine(similarity_top_k=3)


        query_engine_tools = [
            QueryEngineTool.from_defaults( query_engine=engine_AlsaceWine , name="Alsace", description=("Provides info about wines from AlsaceWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_BordeauxWine , name="Bordeaux", description=("Provides info about wines from BordeauxWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_BurgundyWine , name="Burgundy", description=("Provides info about wines from BurgundyWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_ChampagneWine , name="Champagne", description=("Provides info about wines from ChampagneWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_CorsicanWine , name="Corsican", description=("Provides info about wines from CorsicanWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_FrenchWine , name="French", description=("Provides info about wines from FrenchWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_JuraWine , name="Jura", description=("Provides info about wines from JuraWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_RhoneWine , name="Rhone", description=("Provides info about wines from RhoneWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_SavoyWine , name="Savoy", description=("Provides info about wines from SavoyWine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_SouthWestFrenchWine , name="SouthWestFrench", description=("Provides info about wines from SouthWestFrenchWine  region " "Use a detailed plain text question as input to the tool.")),
        ]
        print('returning query_engine_tools')
        return query_engine_tools

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

