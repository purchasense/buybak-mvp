import sys

from fastapi import FastAPI, HTTPException, Body
import asyncio
from fastapi.responses import StreamingResponse
import uuid
import json
import logging
from MyWorkflowFSMC import MyWorkflow
from fastapi.middleware.cors import CORSMiddleware
from buybakworkflow.models import ResearchTopic
from buybakworkflow.MyWorkflowContext import *
from buybakworkflow.rabbitmq import RabbitMQ

import os
from pathlib import Path

os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "/mlruns"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Add handler and formatter if not already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

app = FastAPI()
workflows = {}  # Store the workflow instances in a dictionary


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your Streamlit frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run-buybak-workflow")
async def run_workflow_endpoint(topic: ResearchTopic):

    print(topic)
    fsmc = MyWorkflowContext()
    wf = MyWorkflow(fsmc=fsmc, timeout=2000, verbose=True)
    workflows["0"] = wf  # Store the workflow instance

    async def event_generator():
        loop = asyncio.get_running_loop()
        # yield f"{json.dumps({'workflow': 'MyWorkflow'})}\n\n"

        task = asyncio.create_task(wf.run(user_query=topic.query))
        print(f"event_generator: Created task {task}")
        try:
            async for ev in wf.stream_events():
                print(f"Sending message to frontend: {ev.msg}")
                yield f"{ev.msg}\n\n"
                await asyncio.sleep(0.1)  # Small sleep to ensure proper chunking
            final_result = await task

            final_result_with_url = {
                "result": final_result,
            }

            yield f"{json.dumps({'final_result': final_result_with_url})}\n\n"
        except Exception as e:
            error_message = f"Error in workflow: {str(e)}"
            print(error_message)
            yield f"{json.dumps({'event': 'error', 'message': error_message})}\n\n"
        finally:
            # Clean up
            print("event_generator() finally reached")

    return StreamingResponse(event_generator())


@app.post("/submit-user-input")
async def submit_user_input(topic: ResearchTopic):
    user_input = topic.query
    wf = workflows.get("0")
    if wf and wf.user_input_future:
        loop = wf.user_input_future.get_loop()  # Get the loop from the future
        print(f"submit_user_input: wf.user_input_future loop id {id(loop)}")
        if not wf.user_input_future.done():
            loop.call_soon_threadsafe(wf.user_input_future.set_result, user_input)
            print("submit_user_input: set_result called")
        else:
            print("submit_user_input: future already done")
        return {"status": "input received"}
    else:
        raise HTTPException(
            status_code=404, detail="Workflow not found or future not initialized"
        )

def callback(ch, method, properties, body) -> str:
    print(f"Received {method}, {properties}, {body} ")
    return body

@app.post("/submit-rabbitmq")
async def submit_rabbitmq(topic: ResearchTopic):
    user_input = topic.query
    print(f'submit_rabbitmq: {user_input}')
    rabbitmq = RabbitMQ()
    rabbitmq.publish('buybak', user_input)
    print(f'Sent message: {user_input}')
    response = rabbitmq.consume('buybak_response', callback)
    return {'status': response}


@app.get("/arm_timer")
async def arm_timer():
    wf = workflows.get("0")
    return {"Hello": "World"}

@app.get("/")
async def read_root():
    return {"Hello": "World"}
