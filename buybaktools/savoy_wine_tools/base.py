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


class BuyBakSavoyWineToolSpec(BaseToolSpec):
    """BuyBak French Wines Tool spec."""

    spec_functions = [
        "get_query_engine_tools_for_savoy_wine",
    ]

    def __init__(self):
        print('BBK: Initializing Query Engine Tools for Savoy Wine')
        storage_context_1 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/SavoyWine")

        self.index1 = load_index_from_storage(storage_context_1)

        self.index_loaded = True
        print('------ init savoy_wine query_engine_index done --------')


    def get_query_engine_tools_for_savoy_wine(self) -> []:
        """Query Engine Tools For Savoy Wines"""
        print("Inside get_query_engine_tools_for_savoy_wine")
        engine_SavoyWine =             self.index1.as_query_engine(similarity_top_k=3)


        query_engine_tools = [
            QueryEngineTool.from_defaults( query_engine=engine_SavoyWine , name="Savoy", description=("Provides info about wines from SavoyWine  region " "Use a detailed plain text question as input to the tool.")),
        ]
        print('returning query_engine_tools')
        return query_engine_tools

