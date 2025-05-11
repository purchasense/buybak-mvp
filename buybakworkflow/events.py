from pathlib import Path
from typing import List

from llama_index.core.workflow import Event

from buybakworkflow.schemas import *

class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class ProgressEvent(Event):
    msg: str


