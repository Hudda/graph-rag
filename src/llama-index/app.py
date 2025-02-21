from flask import Flask, request, jsonify

from query import query_engine

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    query_str = request.json['query']

    response = query_engine.query(query_str)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)