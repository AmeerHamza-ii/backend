from flask import Flask, jsonify
from flask_cors import CORS
from model import train_model
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def get_model_output():
    models, mse_values, y_train_values, y_test_values, y_pred_train_values, y_pred_test_values = train_model()

    return jsonify({
        'mse':mse_values, 
        'y_train':y_train_values, 
        'y_test':y_test_values, 
        'y_pred_train':y_pred_train_values, 
        'y_pred_test':y_pred_test_values
    })

if __name__ == '__main__':
    app.run(debug=True)
