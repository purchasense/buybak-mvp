"""
/////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2025 Sameer Kulkarni.
//  BuyBak - Proprietary and Confidencial. All rights reserved.
//
//  author - Sameer Kulkarni, sameer@buybak.io, sameerk1973@gmail.com
//  File   - MyWorkflowEvents.py
//
/////////////////////////////////////////////////////////////////////
                       
This file is AUTO-GENERATED. PLEASE DO NOT EDIT THIS FILE
                       
"""

from buybakworkflow.MyWorkflowContext import *
from llama_index.llms.openai import OpenAI
from buybakworkflow.events import *
from pathlib import Path
from typing import List
from llama_index.core.workflow import Event
from buybakworkflow.schemas import *
from llama_index.core.workflow import ( StartEvent, StopEvent, Workflow, step, Event, Context)
import click
import asyncio
import json
import random
import shutil
import string
import uuid
import json
import asyncio
import inspect
global_list = []


class Booking(BaseModel):
    name:   str = Field(..., description="Name")
    userid:   str = Field(..., description="Unique UserID")
    depart_airport_code:   str = Field(..., description="Departure Airport Code")
    arrival_airport_code:   str = Field(..., description="Arrival Airport Code")
    departure_date:   str = Field(..., description="Depart Date")
    return_date:   str = Field(..., description="Return Date")
    num_adults:   str = Field(..., description="Number of adults")
    num_children:   str = Field(..., description="Number of Children")

class WorkflowStreamingEvent(BaseModel):
    event_type:   Literal["agent", "input"] = Field(..., description="Type of the event")
    event_state:   str = Field(..., description="Finite State")
    event_stimuli:   str = Field(..., description="Stimulus applied to the state transition")
    outline:   str = Field(..., description="Outline")
    message:   str = Field(..., description="Message")

class ResearchTopic(BaseModel):
    query:   str = Field(..., example="example query")



class FirstEvent(Event):
    first_output:   str

class SecondEvent(Event):
    second_output:   str
    response:   str

class WfTimerEvent(Event):
    timer:   str
    name:   str

class TimerFiredEvent(Event):
    timer:   str
    name:   str

class GetUserEvent(Event):
    msg:   str

class SetAIEvent(Event):
    result:   str

class ForecastEvent(Event):
    query:   str

class LiveMarketEvent(Event):
    md:   str

class CompareMarketEvent(Event):
    md:   str

class BuyOrSellEvent(Event):
    md:   str

class FrenchWinesEvent(Event):
    query:   str

class ShoppingCartEvent(Event):
    item:   str


