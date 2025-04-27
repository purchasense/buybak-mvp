import os
import pickle

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

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from helper import get_openai_api_key, get_llama_cloud_api_key
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

index = None
stored_docs = {}
lock = Lock()

index_name = "./saved_index"
pkl_name = "stored_documents.pkl"
# Using AgentQL to crawl a website
agentql_reader = AgentQLWebReader(
    api_key="",  # Replace with your actual API key from https://dev.agentql.com
    params={
        "is_scroll_to_bottom_enabled": True
    },  # Optional additional parameters
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


def insert_url_into_index(url_path):
    """Insert new URL into global index."""
    global index, stored_docs, agentql_reader
    document = agentql_reader.load_data(
        url=url_path,
        query= """
        {
            products[]
            {
                time_depart,
                time_arrive,
                stops,
                total_time,
                airlines,
                airport_codes,
                from_to,
                price
            }
        }
        """
    )

    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    with lock:
        print('------- Regenerating Index from QUERY ----------')
        print(url_path)
        index = VectorStoreIndex(nodes=[], embed_model=embed_model).from_documents(document)
        index.storage_context.persist(persist_dir=index_name)
        print('------- Regeneration DONE --------------------')

    return

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
    manager.register('insert_url_into_index', insert_url_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()
