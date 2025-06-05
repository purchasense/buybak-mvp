from pathlib import Path
from typing import List

from llama_index.core.workflow import Event
from llama_index.core.workflow import Context

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

from typing import Optional, Any, Literal, Dict
from pydantic import BaseModel, Field

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
    llm=OpenAI(model="gpt-4.1"),
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
    llm=OpenAI(model="gpt-4.1"),
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


###############
# Globals
###############
wine_forecast_args: tuple[int, int] = 0,15

buybak_images_str = [
    '/images/Alsace.png',
    '/images/Bordauex-3.jpg',
    '/images/Burgundy-4.png',
    '/images/Champagne-3.png',
    '/images/Corsican.png',
    '/images/SouthWestFrench.png'
    '/images/SouthWestFrench.png'
    '/images/Jura-3.png',
    '/images/Savoy-3.jpg'
]

buybak_wines_str = [
    "Alsace",
    "Bordeaux",
    "Burgundy",
    "Champagne",
    "Corsican",
    "French",
    "SouthWestFrench",
    "Jura",
    "Savoy"
]

wine_forecast =[
    [164.27, 163.64, 163.51, 162.67, 161.94, 161.17, 160.40, 159.82, 159.20, 158.13, 157.74, 157.06, 156.47, 155.88, 154.96],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    []
]

async def makeitem(windex: int, forecast: []) -> str:
    try:
        print(f'makeitem for {windex}, len {len(forecast)}')
        i = random.randint(0, len(forecast)-1)
        print(f'makeitem for {windex} at {i}-------------------> ${forecast[i]} ')
        return forecast[i]

    except Exception as e:
        print(f'Exception {e}')
        print(f'Exception: makeitem -1 -------------------> $200 ')
        return "200"

async def randsleep(caller=None) -> None:
    i = 2
    if caller:
        print(f"{caller} sleeping for {i} seconds.")
    await asyncio.sleep(i)

async def produce(wine: int, q: asyncio.Queue) -> None:
    global wine_forecast

    n = 12
    print(f'produce: {n} times in a loop')
    for x in it.repeat(None, n):  # Synchronous loop for each single producer
        await randsleep(caller=f"Producer {x}")
        price = await makeitem(wine, wine_forecast[wine])
        t = time.perf_counter()
        await q.put((wine, price, t))
        print(f"Producer added ${price} to queue.")
        print(f"---------------------------------------------------------------------")
        await asyncio.sleep(1)

async def consume(name: int, q: asyncio.Queue) -> tuple[str, float]:
        await randsleep(caller=f"Consumer {name}")
        windex, i, t = await q.get()
        now = time.perf_counter()
        print(f"Consumer {name} got element for {windex} at <{i}> in {now-t:0.5f} seconds.")
        q.task_done()
        return windex, i


class MyWorkflowContext():

    live_market_count: int = Field(..., description="Count of live market data events, then stop")
    live_market_data: float = Field(..., description="live market data ")
    live_market_buy: float = Field(..., description="BuyBak Purchase Quantity")
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

    async def generate_stream_event(self, ctx: Context, ev: Event, etype: str, stimulus: str, estate: str, outline: str, msg: str):
        print('generateEvent...')
        ctx.write_event_to_stream( generateEvent(
                etype, 
                estate, 
                stimulus,
                outline,
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
                print(type(obj))
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

        '''
        await self.generate_stream_event(ctx, ev, 
                "timer",
                "WfTimerEvent",
                "TimerState",
                "outline",
                purchase
            )
        '''


    async def suneels_action_function(self,ctx: Context, ev: Event, msg: str):
        print(f"suneels_action_function one_action_1 {msg}")
        quantity = random.randint(1, 6) * 100;
        purchase = "{\"action\": \"Buy\", \"wine\": \"Champagne\", \"quantity\": " + str(quantity) + ", \"price\": 5295900}"
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "ShoppingCart",
                "suneels_action_state",
                "BUY",
                purchase
            )

    async def two_action_1(self,ctx: Context, ev: Event, msg: str):
        print(f"Inside two_action_1 {msg}")

    async def conditional_fore_wine_live(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, msg: str) -> tuple[int, str]:
        print(f"Inside conditional_fore_wine_live {msg} {user_input_future}")

        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_three_action_1() Got user response: {user_response}")

        # Process user_response, which should be a JSON string
        
        '''
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "END_FutureEvent",
                "user_input_state",
                "outline",
                user_response
            )
        '''
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
                "outline",
                user_response
            )
        return False, user_response

    async def conditional_forecast_ema(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, query: str) -> tuple[bool, Any]:
        global wine_forecast, wine_forecast_args

        """Forecast Time Series using the agent workflow"""

        """
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "MLStartEvent",
                "forecast_state",
                "outline",
                "Starting MLForecastor"
            )
        """
        print(query)

        query_prompt = "Query using the buybaktools, forecast time series for next 15 days, and strictly print the LGBMRegressor column as comma separated CSV array"

        timestamp = time.time()
        result, response = self.__iter_over_async_forecaster(query)
        print(f'result: {result}')
        print(f'type: {type(response)}')
        print(f'response.tool_calls: {response.tool_calls}')
        for i in response.tool_calls: 
            if i.tool_kwargs != {}:
                    print(f'---------------> ToolCallResult( {i.tool_kwargs["index"]} {i.tool_kwargs["argin"]})')
                    wine_forecast_args = i.tool_kwargs["index"], i.tool_kwargs["argin"]
                    print(f'wine_forecast_args ---------> {wine_forecast_args}')
        print(f'response: {str(response)}')

        try:
            nospecial = re.sub(r'[^,0-9\.]', '', str(response))
            print(f'nospecial: {nospecial}')
            sdata = nospecial.split(",")
            print(f'sdata: {sdata}')
            float_list = [round(float(num),2) for num in sdata]
            print(f'float_list: {float_list}')
            wine_forecast[wine_forecast_args[0]] = float_list
            for wf in wine_forecast: 
                print(f'-----> ----> ----> : {wf}')

            await self.generate_stream_event(ctx, ev, 
                    "agent",
                    "ForecastEvent",
                    "forecast_state",
                    "outline",
                    str(response)
                )
        except Exception as e:
            print(e)

        return True, f'DONE MLForecastor'

    async def conditional_market_action(self, ctx: Context, ev: Event, live_market_future: asyncio.Future, query: str) -> tuple[bool, Any]:
        global wine_forecast,  buybak_wines_str
        """Live Market Event"""


        consumed = ""
        if self.my_proconq_started == False:
            print(f'Starting my_proconq.....................')
            self.my_queue = asyncio.Queue()
            count = 0
            for i in wine_forecast: 
                if ( len(i) > 5):
                    task = asyncio.create_task(produce(count, self.my_queue))
                    # self.my_producers = self.my_producers.append(task)
                    print(f' tasks[{count}]: started')
                    self.my_proconq_started = True
                    await asyncio.sleep(2)
                count = count + 1
            print(f'Done.....my_proconq.....................')

        print(f'Consuming...')
        windex, price = await consume(self.live_market_count, self.my_queue)
        self.live_market_data = float(price)
        self.live_market_index = windex

        quantity = random.randint(1, 6) * 100;
        price = int(10000 * round(self.live_market_data, 2))

        price_dict = {
            "action": "MD", 
            "wine": buybak_wines_str[self.live_market_index], 
            "quantity": quantity,
            "price": price
        }
        consumed = json.dumps(price_dict)
        print(consumed)

        
        timestamp = time.time()
        await asyncio.sleep(1)

        self.live_market_count = self.live_market_count + 1
        print(f'live_market_count: {self.live_market_count}')

        # if self.live_market_count > 5:
        #     print('Gather my_producers')
        #     # await asyncio.gather(*self.my_producers)
        #     print('Join my_queue')
        #     # await self.my_queue.join()
        #     await self.generate_stream_event(ctx, ev, 
        #         "agent",
        #         "live_market_action",
        #         "StopEvent",
        #         "Done Live Market Events"
        #     )
        #     return False, f'Done Live Market Events'
        # else: 

        await self.generate_stream_event(ctx, ev, 
             "agent",
             "LiveMarketEvent",
             "md_state",
             "outline",
             consumed
        )

        return True, consumed

    async def conditional_compare_market_action(self, ctx: Context, ev: Event, live_market_future: asyncio.Future, md: str) -> tuple[bool, Any]:
        global wine_forecast, buybak_images_str, buybak_wines_str

        """Compare Market Event"""

        await asyncio.sleep(2)

        compared = f'No Comparison Found with {self.live_market_data}'
        try:
            print(f'Comparing wine[{self.live_market_index}] at {(self.live_market_data)} with {(wine_forecast[self.live_market_index])}')
            self.live_market_buy = random.randint(5,28);
            self.live_buybak_alloc = self.live_market_buy * self.live_market_data / 100.0

            if ((self.live_market_data != 0) and (wine_forecast[self.live_market_index][5] - self.live_market_data)) > 0:
                    compared = f'''
                                <table>
                                        
                                    <tr>
                                        <td><img align="top" style={{position:"relative",right:"1px",top:"-30px"}} width="55px" alt={"BuyBak.io"} src={buybak_images_str[self.live_market_index]} /></td>
                                        <td>${self.live_market_data} < forecast by ${round((wine_forecast[self.live_market_index][5] - self.live_market_data), 2)} !!!, </td>
                                    </tr>
                                    <tr>
                                        <td><img align="top" style={{position:"relative",right:"1px",top:"-30px"}} width="55px" alt={"BuyBak.io"} src="/images/BuyBakGreenLogoPitchDeck.png" /></td>
                                        <td>BuyBak Alloc {self.live_market_buy/100.0} for ${round(self.live_buybak_alloc, 2)}</td>
                                    </tr>
                                </table>'''

                    await self.generate_stream_event(ctx, ev, 
                        "agent",
                        "CompareMarketEvent",
                        "compare_markets",
                        "outline",
                        compared
                    )
                    return True, compared
            else:
                    compared = f'{(self.live_market_data)} Greater than FC by {round((wine_forecast[self.live_market_index][5] - self.live_market_data), 2)}, <span>&nbsp;&nbsp;&nbsp;</span>ignoring... '
                    await self.generate_stream_event(ctx, ev, 
                        "agent",
                        "CompareMarketEvent",
                        "compare_markets",
                        "outline",
                        compared
                    )
                    return False, compared
        except Exception as e:
            print(e)
            return False, compared

    async def conditional_buy_or_sell_action(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, md: str) -> tuple[bool, str]:
        global wine_forecast, buybak_wines_str

        print(f"Inside conditional_buy_or_sell {md} {user_input_future}")
        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_buy_or_sell() Got {user_response}")
        
        resp = f'<html><body>User decided to <font style="fontWeight: \"bold\", color: \"blue\" ">{user_response}</font></body></html>'

        # Process user_response, which should be a JSON string
        await self.generate_stream_event(ctx, ev, 
                "agent",
                "DecisionEvent",
                "user_input_state",
                "outline",
                resp
            )

        price = int(10000 * round(self.live_market_data, 2))
        item_dict = {
            "action": "Buy", 
            "wine": buybak_wines_str[self.live_market_index], 
            "quantity": self.live_market_buy,
            "price": price
        }
        item_json_str = json.dumps(item_dict)
        print(item_json_str)

        if "Buy" in user_response:
            # we have a buyer.
            return 0, item_json_str
        elif "No" in user_response:
            # we continue with more
            return 1, item_json_str
        else:
            return 2, item_json_str

    async def buybak_french_wines_action(self,ctx: Context, ev: Event, query: str):

        """French Wines ToolCall using the agent workflow"""

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "FrenchWinesEvent",
                "french_wines_state",
                "outline",
                "Querying...",
            )
        print(query)

        # query_prompt = "What are the different regions of french wines? Print result strictly as HTML Table"

        timestamp = time.time()
        result, response = self.__iter_over_async_forecaster(query)
        print(f'result: {result}')
        print(f'response: {response}')

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "FrenchWinesEvent",
                "french_wine_state",
                "outline",
                str(response)
            )

    async def shopping_cart_action(self,ctx: Context, ev: Event, item: str) -> str:

        """ Shopping Cart """

        outcome = f'S-Cart {item}'
        print(outcome)

        await self.generate_stream_event(ctx, ev, 
                "agent",
                "ShoppingCartEvent",
                "shopping_cart_state",
                "BUY",
                item
            )
        print(f'shopping_cart_action: {item}')
        return item


if __name__ == "__main__":
    import sys
    print("initializing MyWorkflowContext...")

    query = "Get BuyBak French Wines List as JSON"
    if len(sys.argv) == 2:
        query = argv[1]
        
    print(f'query: {query}')

    fsmcontext = MyWorkflowContext()
    ev = Event()
    ctx = Context(fsmcontext)
    user_input_future = asyncio.Future()
    retval, response = fsmcontext.conditional_forecast_ema(ctx, ev, user_input_future, "Sameer Kulkarni")
    print(response)

