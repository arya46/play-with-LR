from flask import Flask, request, render_template, url_for
from flask_cors import CORS
import json
import numpy as np

from utils.helper_functions import generate_random_data, logRegClassifier, standardize_data

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/api/generate-data')
def generate_data():
    X, Y = generate_random_data()
    return json.dumps({
        'x1': X[:,0],
        'x2': X[:,1],
        'y': Y
    }, cls=NumpyEncoder)

@app.route('/api/train-data', methods=['POST'])
def train_data():

    data_x = np.column_stack((request.json['dataset']['x1'], request.json['dataset']['x2']))
    data_y = list(request.json['dataset']['y'])

    data_X, STD, MEAN = standardize_data(data_x)
    return json.dumps(logRegClassifier(data_X, data_y, \
        request.json['learning_rate'], request.json['max_iter'], request.json['C'], \
            STD, MEAN))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":

    # RUNNNING FLASK APP    
    app.run(debug=True, host = '0.0.0.0', port=8080)