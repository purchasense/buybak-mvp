from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
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
from llama_index.llms.openai import OpenAI

from buybakworkflow.events import *

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

class MyWorkflow(Workflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize the Future
        self.user_input_future = asyncio.Future()
        print('MyWorkflow: __init__')

    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop() # store the event loop
        return await super().run(*args, **kwargs)

    async def reset_user_input_future(self):
        self.user_input_future = self.loop.create_future()
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream( generateEvent( 
                "agent", 
                inspect.currentframe().f_code.co_name, 
                "StartEvent", 
                "outline", 
                "Inside step_one"
            )
        )
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        ctx.write_event_to_stream( generateEvent( 
                "agent", 
                inspect.currentframe().f_code.co_name, 
                "FirstEvent", 
                "outline", 
                "Inside step_two"
            )
        )
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response="step_two completed"
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> GetUserEvent:
        ctx.write_event_to_stream( generateEvent( 
                "input", 
                inspect.currentframe().f_code.co_name, 
                "SecondEvent", 
                "outline", 
                "Do you approve this outline? If not, please provide feedback."
            )
        )

        return GetUserEvent(
            msg="Look for input from user"
        )

    @step
    async def step_four(self, ctx: Context, ev: GetUserEvent) -> StopEvent:
        print(ev.msg)
        # Initialize the future if it's None
        if self.user_input_future is None:
            print( "self.user_input_future is None, initializing user input future")
            self.user_input_future = self.loop.create_future()

        if not self.user_input_future.done():
            print(f"step_three() Event loop id {id(self.loop)}, waiting for user_input...")
            user_response = await self.user_input_future
            print(f"step_three(): Got user response: {user_response}")
            self.user_input_future = self.loop.create_future()

        # Process user_response, which should be a JSON string
        ctx.write_event_to_stream( generateEvent( 
                "agent", 
                inspect.currentframe().f_code.co_name, 
                "StopEvent", 
                "outline", 
                "Finally Done."
            )
        )
        return StopEvent(result=user_response)

async def run_workflow(first_input: str):
    w = MyWorkflow(timeout=30, verbose=True)
    result = await w.run(first_input=first_input)
    print("Final result", result)

@click.command()
@click.option(
    "--user-query",
    "-q",
    required=False,
    help="The user query",
    default="StreamingEventsWorkflow",
)
def main(user_query: str):
    asyncio.run(run_workflow(user_query))

if __name__ == "__main__":
    asyncio.run(main())
