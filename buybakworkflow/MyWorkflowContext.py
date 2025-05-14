from pathlib import Path
from typing import List

from llama_index.core.workflow import Event
from llama_index.core.workflow import Context


from buybakworkflow.schemas import *

class MyWorkflowContext():

    def __init__(self,*args,**kwargs):
        print("Inside __init__")

    async def action_1(self,ctx: Context):
        print("Inside action_1")

    async def action_2(self,ctx: Context):
        print("Inside action_1")

    async def action_3(self,ctx: Context):
        print("Inside action_1")
