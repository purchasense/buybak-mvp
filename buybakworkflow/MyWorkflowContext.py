from pathlib import Path
from typing import List

from llama_index.core.workflow import Event
from llama_index.core.workflow import Context

from buybakworkflow.schemas import *
import time
import itertools as it
import random
import inspect
import threading
import json
import asyncio
import re
import string
import csv
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent

from buybaktools.time_series_tools import BuyBakTimeSeriesToolSpec, BuyBakLineItemAndIndicators, BuyBakTimeSeries
from buybaktools.french_wines_tools import BuyBakFrenchWinesToolSpec


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

system_prompt = f"""You are now connected to the BuyBakTimeSeries Tools, that 1. predicts the EMA values for a live array of [['open','high','low','close']], and 2. forecasts future EMA using the MLForecaster.
Only enter details that the user has explicitly provided. Return the value from the tools provided for the predict method.
Do not make up any details.
"""
buybak_time_series_tools = BuyBakTimeSeriesToolSpec().to_tool_list()
buybak_ts_agent = FunctionAgent(
    name="BuyBakTimeSeriesAgent",
    description="Booking agent that predicts the next EMA values from a time series AND Provides info about wines from different region in France",
    tools=buybak_time_series_tools,
    llm=OpenAI(model="gpt-4o"),
    system_prompt=system_prompt,
    verbose=True,
)

system_prompt_french_wines = f"""You are now connected to the BuyBakFrenchWines Tools, that uses storage for 10 different wines, and invokes the list of query_tool, one per wine district, to answer user queries. Return the value strictly as an HTML output.
Do not make up any details.
"""
buybak_french_wines_tools = BuyBakFrenchWinesToolSpec().to_tool_list()
buybak_french_wines_agent = FunctionAgent(
    name="BuyBakFrenchWinesAgent",
    description="Agent that tablulates French Wines from the toolspec query_engine_tool loaded from the storage",
    tools=buybak_french_wines_tools,
    llm=OpenAI(model="gpt-4o"),
    system_prompt=system_prompt_french_wines,
    verbose=True,
)

agent_workflow = AgentWorkflow(
    agents=[buybak_ts_agent, buybak_french_wines_agent],
    root_agent=buybak_ts_agent.name,
    initial_state={
    },
    verbose=True,
)


def generateEvent(etype: str, estate: str, stimuli: str, outline: str, message: str) -> Event:
        return Event(
                msg=json.dumps(
                    {
                        "event_type": etype,
                        "event_state": estate,
                        "event_stimuli": stimuli,
                        "event_content": {
                            "outline": outline,
                            "message": message
                        },
                    }
                )
            )


async def makeitem(forecast: []) -> str:
    i = random.randint(0, 10)
    if len(forecast) > 9:
        print(f'make_item {i} -------------------> {forecast[i]} ')
        return forecast[i]
    else: 
        return 200.0

async def randsleep(caller=None) -> None:
    i = random.randint(0, 10)
    if caller:
        print(f"{caller} sleeping for {i} seconds.")
    await asyncio.sleep(i)

async def produce(q: asyncio.Queue) -> None:
    n = 6
    print(f'produce: {n} times in a loop')
    forecast = [164.27, 163.64, 163.51, 162.67, 161.94, 161.17, 160.40, 159.82, 159.20, 158.13, 157.74, 157.06, 156.47, 155.88, 154.96]
    for x in it.repeat(None, n):  # Synchronous loop for each single producer
        await randsleep(caller=f"Producer {x}")
        i = await makeitem(forecast)
        t = time.perf_counter()
        await q.put((i, t))
        print(f"Producer added <{i}> to queue.")
        print(f"---------------------------------------------------------------------")
        await asyncio.sleep(1)

async def consume(name: int, q: asyncio.Queue) -> str:
        await randsleep(caller=f"Consumer {name}")
        i, t = await q.get()
        now = time.perf_counter()
        print(f"Consumer {name} got element <{i}> in {now-t:0.5f} seconds.")
        q.task_done()
        return f'{i}'


class MyWorkflowContext():

    live_market_count: int = Field(..., description="Count of live market data events, then stop")
    live_market_data: float = Field(..., description="live market data ")
    live_market_forecast = []
    my_proconq_started: bool = Field(..., description="ProConQ started")
    my_queue: asyncio.Queue
    my_producers: Any
    my_consumers: Any


    def __init__(self,*args,**kwargs):
        print("Inside __init__")
        random.seed(444)
        self.live_market_count = 0
        self.live_market_data = 0
        self.my_proconq_started = False

    async def generate_stream_event(self, ctx: Context, ev: Event, etype: str, stimulus: str, estate: str, msg: str):
        print('generateEvent...')
        ctx.write_event_to_stream( generateEvent(
                etype, 
                estate, 
                stimulus,
                "outline",
                msg, 
            )
        )

    async def slow_forecast_time_series(self, query_prompt: str) -> tuple[bool, Any]:
        """Forecast Time Series using the agent workflow"""
        global index, stored_docs, agentql_reader, buybak_time_series_tools

        print('------- QUERY FORECASTING ----------')
        ctx = Context(agent_workflow)
        response = await agent_workflow.run(query_prompt)
        print(response)
        print('------- Prediction DONE --------------------')

        return True, response

    def __iter_over_async_forecaster(self, query_prompt: str):
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
                obj = await self.slow_forecast_time_series(query_prompt)
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

    async def one_action_1(self,ctx: Context, ev: Event, msg: str):
        print(f"Inside one_action_1 {msg}")

    async def armTimer(self, ctx: Context, ev: Event, timer: str, msg: str):
        print(f"Inside armTimer {timer}, {msg}")
        await asyncio.sleep(int(timer))

        await self.generate_stream_event(ctx, ev, 
                "timer",
                "WfTimerEvent",
                "TimerState",
                "Timer fired!"
            )


    async def suneels_action_function(self,ctx: Context, ev: Event, msg: str):
        print(f"suneels_action_function one_action_1 {msg}")

    async def two_action_1(self,ctx: Context, ev: Event, msg: str):
        print(f"Inside two_action_1 {msg}")

    async def conditional_fore_wine_live(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, msg: str) -> tuple[int, str]:
        print(f"Inside conditional_fore_wine_live {msg} {user_input_future}")

        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_three_action_1() Got user response: {user_response}")

        # Process user_response, which should be a JSON string
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "END_FutureEvent",
                "user_input_state",
                user_response
            )
        if "AI" in user_response:
            return 0, user_response
        elif "wines" in user_response:
            return 1, user_response
        else:
            return 2, user_response

    async def conditional_four_action_1(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, msg: str) -> tuple[bool, str]:
        print(f"Inside conditional_FOUR_action_1 {msg} {user_input_future}")
        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_four_action_1() Got user response: {user_response}")

        # Process user_response, which should be a JSON string
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "END_FutureEvent",
                "user_input_state",
                user_response
            )
        return False, user_response

    async def conditional_forecast_ema(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, query: str) -> tuple[bool, Any]:

        """Forecast Time Series using the agent workflow"""

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "ForecastEvent",
                "forecast_state",
                "Starting MLForecastor"
            )
        print(query)

        query_prompt = "Query using the buybaktools, forecast time series for next 15 days, and strictly print the LGBMRegressor column as comma separated CSV array"

        timestamp = time.time()
        result, response = self.__iter_over_async_forecaster(query)
        print(f'result: {result}')
        print(f'response: {response}')
        print(f'response: {str(response)}')

        nospecial = re.sub(r'[^,0-9\.]', '', str(response))
        print(f'nospecial: {nospecial}')
        sdata = nospecial.split(",")
        print(f'sdata: {sdata}')
        float_list = [float(num) for num in sdata]
        print(f'float_list: {float_list}')
        self.live_market_forecast = float_list

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "ForecastEvent",
                "forecast_state",
                str(response)
            )

        return True, f'DONE MLForecastor'

    async def conditional_market_action(self, ctx: Context, ev: Event, live_market_future: asyncio.Future, query: str) -> tuple[bool, Any]:

        """Live Market Event"""


        consumed = ""
        if self.my_proconq_started == False:
            print(f'Starting my_proconq.....................')
            self.my_queue = asyncio.Queue()
            self.my_producers = [asyncio.create_task(produce(self.my_queue))]
            self.my_proconq_started = True
        else:
            print(f'Consuming...')
            consumed = await consume(self.live_market_count, self.my_queue)
            self.live_market_data = float(consumed)
        
        timestamp = time.time()
        await asyncio.sleep(1)

        self.live_market_count = self.live_market_count + 1
        print(f'live_market_count: {self.live_market_count}')

        if self.live_market_count > 5:
            print('Gather my_producers')
            # await asyncio.gather(*self.my_producers)
            print('Join my_queue')
            # await self.my_queue.join()
            await self.generate_stream_event(ctx, ev, 
                "agent",
                "live_market_action",
                "StopEvent",
                "Done Live Market Events"
            )
            return False, f'Done Live Market Events'
        else: 
            print(f'Now consuming {consumed}')
            '''
            await self.generate_stream_event(ctx, ev, 
                "agent",
                "LiveMarketEvent",
                "live_market_action",
                consumed
            )
            '''
            return True, consumed

    async def conditional_compare_market_action(self, ctx: Context, ev: Event, live_market_future: asyncio.Future, md: str) -> tuple[bool, Any]:

        """Compare Market Event"""

        await asyncio.sleep(2)

        print(f'Comparing {self.live_market_data} with {self.live_market_forecast}')
        compared = f'No Comparison Found with {self.live_market_data}'

        if ((self.live_market_data != 0) and (self.live_market_forecast[5] - self.live_market_data)) > 0:
                compared = f'{self.live_market_data} Less Than Forecast by {round((self.live_market_forecast[5] - self.live_market_data), 2)}!!!'
                await self.generate_stream_event(ctx, ev, 
                    "agent",
                    "CompareMarketEvent",
                    "compare_markets",
                    compared
                )
                return True, compared
        else:
                compared = f'{self.live_market_data} Greater than FC by {round((self.live_market_forecast[5] - self.live_market_data), 2)}, next... '
                await self.generate_stream_event(ctx, ev, 
                    "agent",
                    "CompareMarketEvent",
                    "compare_markets",
                    compared
                )
                return False, compared

    async def conditional_buy_or_sell_action(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, md: str) -> tuple[bool, str]:
        print(f"Inside conditional_buy_or_sell {md} {user_input_future}")
        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_buy_or_sell() Got {user_response}")

        # Process user_response, which should be a JSON string
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "END_FutureEvent",
                "user_input_state",
                user_response
            )

        if "Buy" in user_response:
            return True, user_response
        else:
            return False, user_response

    async def buybak_french_wines_action(self,ctx: Context, ev: Event, query: str) -> tuple[bool, Any]:

        """French Wines ToolCall using the agent workflow"""

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "FrenchWinesEvent",
                "french_wines_state",
                "Querying...",
            )
        print(query)

        query_prompt = "What are the different regions of french wines? Print result strictly as HTML Table"

        timestamp = time.time()
        result, response = self.__iter_over_async_forecaster(query)
        print(f'result: {result}')
        print(f'response: {response}')

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "FrenchWinesEvent",
                "french_wine_state",
                str(response)
            )


if __name__ == "__main__":
    print("initializing MyWorkflowContext...")
    fsmcontext = MyWorkflowContext()
    ev = Event()
    ctx = Context(fsmcontext)
    user_input_future = asyncio.Future()
    retval, response = fsmcontext.conditional_forecast_ema(ctx, ev, user_input_future, "Sameer Kulkarni")
    print(response)

