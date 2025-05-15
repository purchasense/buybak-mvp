from pathlib import Path
from typing import List

from llama_index.core.workflow import Event
from llama_index.core.workflow import Context

from buybakworkflow.schemas import *
import inspect
import json
import asyncio

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

class MyWorkflowContext():

    def __init__(self,*args,**kwargs):
        print("Inside __init__")

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

    async def one_action_1(self,ctx: Context, ev: Event, msg: str):
        print(f"Inside one_action_1 {msg}")

    async def suneels_action_function(self,ctx: Context, ev: Event, msg: str):
        print(f"suneels_action_function one_action_1 {msg}")

    async def two_action_1(self,ctx: Context, ev: Event, msg: str):
        print(f"Inside two_action_1 {msg}")

    async def conditional_three_action_1(self,ctx: Context, ev: Event, user_input_future: asyncio.Future, msg: str) -> tuple[bool, str]:
        print(f"Inside conditional_three_action_1 {msg} {user_input_future}")
        ctx.write_event_to_stream( generateEvent(
                "input",
                "conditional_state",
                "StartFutureEvent",
                "outline",
                "Please enter INPUT"
            )
        )
        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_three_action_1() Got user response: {user_response}")

        # Process user_response, which should be a JSON string
        ctx.write_event_to_stream( generateEvent(
                "agent",
                "user_input_state",
                "END_FutureEvent",
                "outline",
                user_response
            )
        )
        if "AI" in user_response:
            return True, user_response
        else:
            return False, user_response

    async def conditional_four_action_1(self,ctx: Context, ev: Event, user_input_future: asyncio.Future,msg: str) -> tuple[bool, str]:
        print(f"Inside conditional_FOUR_action_1 {msg} {user_input_future}")
        ctx.write_event_to_stream( generateEvent(
                "input",
                "conditional_state",
                "StartFutureEvent",
                "outline",
                "Please enter INPUT"
            )
        )
        if not user_input_future.done():
            print(f"waiting for user_input...")
            user_response = await user_input_future
            print(f"conditional_four_action_1() Got user response: {user_response}")

        # Process user_response, which should be a JSON string
        ctx.write_event_to_stream( generateEvent(
                "agent",
                "user_input_state",
                "END_FutureEvent",
                "outline",
                user_response
            )
        )
        return False, user_response
