#!/bin/bash

# start backend index server
python ./index_server.py &
echo "index_server running..."

# wait for the server to start - if creating a brand new huge index, on startup, increase this further
sleep 10

# start the flask server
python ./flask_demo.py &

echo "Building and running React Frontend..."
# assumes you've ran npm install already (dockerfile does this during build)
cd react_frontend && npm run build && serve -s build
