#!/bin/bash
set -x
/usr/local/bin/python3 FSMC/FSMC.py  FSMC/MyWorkflow.sm  > MyWorkflowFSMC.py
poetry run uvicorn MyWorkflowServer:app --host 0.0.0.0 --port 8080
