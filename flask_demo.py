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
manager.register('insert_url_into_index')
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


@app.route("/query_new_url", methods=["GET"])
def query_new_URL():
    global manager
    query_url = request.args.get("text", None)
    if query_url is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    print(query_url)
    response = manager.insert_url_into_index(query_url)._getvalue()
    print(response)
    response_json = {
        "text": str(response),
        "sources": [{"text": "query_new_url",
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
