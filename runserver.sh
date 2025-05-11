#!/bin/bash

poetry run uvicorn StreamingServer:app --host 0.0.0.0 --port 8080
