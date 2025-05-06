import asyncio
import os
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# initialize manager connection
# NOTE: you might want to handle the password in a less hardcoded way
manager = BaseManager(('', 5602), b'password')
manager.register('query_index')
manager.register('update_time_series')
manager.register('get_documents_list')
manager.connect()


@app.route("/query", methods=["GET"])
def query_index():
    global manager
    query_text = request.args.get("text", None)
    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    response = manager.query_index(query_text)._getvalue()
    print(response)
    print('------------------------------------------')
    response_json = {
        "text": str(response),
        "sources": [{"text": str(x.text), 
                     "similarity": round(x.score, 2),
                     "doc_id": str(x.id_),
                     "start": "",
                     "end": "",
                    } for x in response.source_nodes]
    }
    return make_response(jsonify(response_json)), 200

# query_prompt = "Convert the following list into a BuyBakTimeSeries. Then call the predict tool to find the answer. [[143.39, 154.93, 142.66, 149.24], [153.57, 154.44, 145.21, 146.58], [146.33, 161.87, 145.81, 161.06], [158.76, 160.03, 152.2, 155.37], [155.59, 159.86, 155.59, 159.4], [162.31, 164.03, 159.92, 161.47], [161.57, 162.05, 157.65, 158.68], [155.47, 158.18, 153.91, 155.5], [156.61, 157.07, 150.9, 153.36], [150.96, 151.06, 148.4, 149.86], [151.07, 154.61, 150.87, 153.9], [157.91, 160.02, 156.35, 157.72], [158.52, 161.71, 158.09, 161.47], [167.1, 168.24, 163.0, 163.85], [164.26, 164.95, 160.38, 162.42], [162.04, 162.68, 159.39, 162.06], [159.86, 161.37, 157.15, 160.89], [162.52, 163.94, 160.93, 162.79], [164.99, 166.45, 163.66, 165.76]]. Finally, call the MSE on the predicted sample"


@app.route("/update_time_series", methods=["POST"])
def update_time_series():
    global manager
    print(request.get_data())
    query_prompt = request.get_data()
    if query_prompt is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    print(str(query_prompt))

    response = manager.update_time_series(str(query_prompt))._getvalue()
    response_json = {
        "text": str("SUCCESS"),
        "sources": [{"text": "update_time_series",
                     "similarity": 0.0,
                     "doc_id": 1,
                     "start": "",
                     "end": "",
                    }]
    }
    return make_response(jsonify(response_json)), 200


@app.route("/getDocuments", methods=["GET"])
def get_documents():
    document_list = manager.get_documents_list()._getvalue()

    return make_response(jsonify(document_list)), 200
    

@app.route("/")
def home():
    return "Hello, World! Welcome to the llama_index docker image!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)
