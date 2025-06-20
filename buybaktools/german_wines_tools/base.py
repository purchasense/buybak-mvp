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

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool


class BuyBakGermanWinesToolSpec(BaseToolSpec):
    """BuyBak German Wines Tool spec."""

    spec_functions = [
        "get_query_engine_tools_for_german_wines",
    ]

    def __init__(self):
        print('BBK: Initializing Query Engine Tools for German Wines')
        storage_context_1 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/AhrWines")
        storage_context_2 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/BadenWines")
        storage_context_3 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/FranconiaWines")
        storage_context_4 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/HessischeWines")
        storage_context_5 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/KaiserstuhlWines")
        storage_context_6 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/GermanWines")
        storage_context_7 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/MoselWines")
        storage_context_8 = StorageContext.from_defaults( persist_dir="./TradesCSV/storage/german/RheinhessenWines")

        self.index1 = load_index_from_storage(storage_context_1)
        self.index2 = load_index_from_storage(storage_context_2)
        self.index3 = load_index_from_storage(storage_context_3)
        self.index4 = load_index_from_storage(storage_context_4)
        self.index5 = load_index_from_storage(storage_context_5)
        self.index6 = load_index_from_storage(storage_context_6)
        self.index7 = load_index_from_storage(storage_context_7)
        self.index8 = load_index_from_storage(storage_context_8)

        self.index_loaded = True
        print('------ init german wines query_engine_index done --------')


    def get_query_engine_tools_for_german_wines(self) -> []:
        """Extract the BuyBakTimeSeries structure from the input """
        print("Inside get_query_engine_tools_for_german_wines")
        engine_AhrWine =             self.index1.as_query_engine(similarity_top_k=3)
        engine_BadenWine =           self.index2.as_query_engine(similarity_top_k=3)
        engine_FranconiaWine =           self.index3.as_query_engine(similarity_top_k=3)
        engine_HessischeWine =          self.index4.as_query_engine(similarity_top_k=3)
        engine_KaiserstuhlWine =           self.index5.as_query_engine(similarity_top_k=3)
        engine_GermanWine =             self.index6.as_query_engine(similarity_top_k=3)
        engine_MoselWine =               self.index7.as_query_engine(similarity_top_k=3)
        engine_RheinhessenWine =              self.index8.as_query_engine(similarity_top_k=3)


        query_engine_tools = [
            QueryEngineTool.from_defaults( query_engine=engine_AhrWine , name="Ahr", description=("Provides info about wines from Ahr Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_BadenWine , name="Baden", description=("Provides info about wines from Baden Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_FranconiaWine , name="Franconia", description=("Provides info about wines from Franconia Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_HessischeWine , name="Hessische", description=("Provides info about wines from Hessische Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_KaiserstuhlWine , name="Kaiserstuhl", description=("Provides info about wines from Kaiserstuhl Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_GermanWine , name="German", description=("Provides info about wines from German  Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_MoselWine , name="Mosel", description=("Provides info about wines from Mosel Wine  region " "Use a detailed plain text question as input to the tool.")),
            QueryEngineTool.from_defaults( query_engine=engine_RheinhessenWine , name="Rheinhessen", description=("Provides info about wines from Rheinhessen Wine  region " "Use a detailed plain text question as input to the tool.")),
        ]
        print('returning query_engine_tools')
        return query_engine_tools


