import os
import pickle
import asyncio
import threading
from typing import Any

import os, json
import pandas as pd
import matplotlib.pyplot as plt
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import AgentQLWebReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from helper import get_openai_api_key, get_llama_cloud_api_key, get_tavily_api_key
from typing import Optional
from typing import ClassVar
from typing import List
from pydantic import BaseModel
import json
from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.schema import Document
from llama_index.core.service_context import ServiceContext


from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
)
from llama_index.readers.web import AgentQLWebReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from buybaktools.time_series_tools import BuyBakTimeSeriesToolSpec, BuyBakLineItemAndIndicators, BuyBakTimeSeries
from llama_index.core.tools import FunctionTool

from llama_index.core.agent.workflow import FunctionAgent
import os

predictions = {"list_items": []}
forecastors = {"list_items": []}

index = None
stored_docs = {}
lock = Lock()

import nest_asyncio
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()
tavily_api_key = get_tavily_api_key()

print(f'LLAMA  {llama_cloud_api_key}')
print(f'OPENAI {openai_api_key}')
print(f'TAVILY {tavily_api_key}')



index_name = "./saved_index"
pkl_name = "stored_documents.pkl"
# Using AgentQL to crawl a website
agentql_reader = AgentQLWebReader(
    api_key="",  # Replace with your actual API key from https://dev.agentql.com
    params={
        "is_scroll_to_bottom_enabled": True
    },  # Optional additional parameters
)
system_prompt = f"""You are now connected to the BuyBakTimeSeries Tools, that 1. predicts the EMA values for a live array of [['open','high','low','close']], and 2. forecasts future EMA using the MLForecaster.
Only enter details that the user has explicitly provided. Return the value from the tools provided for the predict method.
Do not make up any details.
"""
buybak_time_series_tools = BuyBakTimeSeriesToolSpec().to_tool_list()
buybak_ts_agent = FunctionAgent(
    name="BuyBakTimeSeriesAgent",
    description="Booking agent that predicts the next EMA values from a time series",
    tools= buybak_time_series_tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt=system_prompt,
    verbose=True,
)

agent_workflow = AgentWorkflow(
    agents=[buybak_ts_agent],
    root_agent=buybak_ts_agent.name,
    initial_state={
    },
    verbose=True,
)


def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs, agentql_reader
    
    with lock:

        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        data = SimpleDirectoryReader(input_dir="./TradesCSV/").load_data(show_progress=False)
        index = VectorStoreIndex.from_documents(data)


def query_index(query_text):
    """Query the global index."""
    global index
    # Load documents from a single page URL
    print(query_text)
    llm = OpenAI(model="gpt-4o-mini")
    response = index.as_query_engine(
        similarity_top_k=5,
        llm=llm,
    ).query(query_text)
    return response


# SAMPLE query        "Convert the following list into a BuyBakTimeSeries, with 'line_items', in the json argument. Then call the predict tool to find the answer. [[143.39, 154.93, 142.66, 149.24], [153.57, 154.44, 145.21, 146.58], [146.33, 161.87, 145.81, 161.06], [158.76, 160.03, 152.2, 155.37], [155.59, 159.86, 155.59, 159.4], [162.31, 164.03, 159.92, 161.47], [161.57, 162.05, 157.65, 158.68], [155.47, 158.18, 153.91, 155.5], [156.61, 157.07, 150.9, 153.36], [150.96, 151.06, 148.4, 149.86], [151.07, 154.61, 150.87, 153.9], [157.91, 160.02, 156.35, 157.72], [158.52, 161.71, 158.09, 161.47], [167.1, 168.24, 163.0, 163.85], [164.26, 164.95, 160.38, 162.42], [162.04, 162.68, 159.39, 162.06], [159.86, 161.37, 157.15, 160.89], [162.52, 163.94, 160.93, 162.79], [164.99, 166.45, 163.66, 165.76]]. Finally, call the MSE on the predicted sample")

async def slow_update_time_series(query_prompt) -> tuple[bool, Any]:
    """Insert new URL into global index."""
    global index, stored_docs, agentql_reader, buybak_time_series_tools
    
    print('------- QUERY PREDICTION ----------')
    ctx = Context(agent_workflow)
    response = await agent_workflow.run(query_prompt)
    print(response)
    print('------- Prediction DONE --------------------')

    return True, response

def __iter_over_async(query_prompt: str):
    """
    Iterates over the async iterable and yields formatted chunks.

    Yields:
        str: Formatted chunk of response text.
    """
    print('----- __iter_over_async ....')
    print(query_prompt)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def get_next(query_prompt) -> tuple[bool, Any]:
        """
        Retrieves the next chunk from the iterator.

        Returns:
            tuple[bool, Any]: A tuple with a boolean indicating if the iteration is done and the chunk.
        """
        try:
            print('----- try await slow....')
            print(query_prompt)
            obj = await slow_update_time_series(query_prompt)
            print('-------- return slow_update response')
            print(obj)
            return True, obj
        except StopAsyncIteration:
            return True, obj
        except Exception as e:
            print(f"Error in get_next: {e}")
            return True, f"{e}"

    try:
        while True:
            print('----- while_true ....')
            print(query_prompt)
            done, obj = loop.run_until_complete(get_next(query_prompt))
            if done:
                break
    finally:
        loop.close()
    return obj

def update_time_series(query_prompt):
    with lock:
        import time
        timestamp = time.time()
        result, response = __iter_over_async(query_prompt)
        print(f'result: {result}')
        print(f'response: {response}')
        predictions["list_items"].append({str(int(timestamp)): f'{response}'})
        print(predictions)
        return f'{response}'
            
async def slow_forecast_time_series(query_prompt: str) -> tuple[bool, Any]:
    """Forecast Time Series using the agent workflow"""
    global index, stored_docs, agentql_reader, buybak_time_series_tools
    
    print('------- QUERY FORECASTING ----------')
    ctx = Context(agent_workflow)
    response = await agent_workflow.run(query_prompt)
    print(response)
    print('------- Prediction DONE --------------------')

    return True, response

def __iter_over_async_forecaster(query_prompt: str):
    """
    Iterates over the async iterable and yields formatted chunks.

    Yields:
        str: Formatted chunk of response text.
    """
    print('----- __iter_over_async ....')
    print(query_prompt)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def get_next(query_prompt) -> tuple[bool, Any]:
        """
        Retrieves the next chunk from the iterator.

        Returns:
            tuple[bool, Any]: A tuple with a boolean indicating if the iteration is done and the chunk.
        """
        try:
            print('----- try await slow....')
            print(query_prompt)
            obj = await slow_forecast_time_series(query_prompt)
            print('-------- return slow_update response')
            print(obj)
            return True, obj
        except StopAsyncIteration:
            return True, obj
        except Exception as e:
            print(f"Error in get_next: {e}")
            return True, f"{e}"

    try:
        while True:
            print('----- while_true ....')
            print(query_prompt)
            done, obj = loop.run_until_complete(get_next(query_prompt))
            if done:
                break
    finally:
        loop.close()
    return obj

def forecast_time_series(query_prompt):
    with lock:
        import time
        timestamp = time.time()
        result, response = __iter_over_async_forecaster(query_prompt)
        print(f'result: {result}')
        print(f'response: {response}')
        forecastors["list_items"].append({str(int(timestamp)): f'{response}'})
        print(forecastors)
        return f'{response}'
            

def get_predictions():
    """Get the Notifications to the frontend periodically"""
    global predictions

    local_predictions = {}
    with lock:
        local_predictions = predictions
        
    print(f'local_type {type(local_predictions)}')
    return local_predictions
    
def get_forecastors():
    """Get the Notifications to the frontend periodically"""
    global forecastors

    local_forecastors = {}
    with lock:
        local_forecastors = forecastors
        
    print(f'local_type {type(local_forecastors)}')
    return local_forecastors
    
def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('update_time_series', update_time_series)
    manager.register('forecast_time_series', forecast_time_series)
    manager.register('get_documents_list', get_documents_list)
    manager.register('get_predictions', get_predictions)
    manager.register('get_forecastors', get_forecastors)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()
