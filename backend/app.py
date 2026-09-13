# Import data manipulation libraries
import numpy as np
import pandas as pd

# For serialization
import joblib

# Flask API
from flask import Flask, request, jsonify

# Import logging
import logging
import sys

# Initialize the Flask app with a name
superkart_api = Flask("superkart_sales_app")

# Debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Module name: {__name__}")
logger.info(f"Flask app name: {superkart_api.name}")
logger.info(f"Root path: {superkart_api.root_path}")

# Load the trained sales prediction model
model = joblib.load("sales_price_prediction_model_v1_0.joblib")


# Features expected by the model
FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Store_Age_Years',
    'Product_Type_Category',
    'Product_Id_char'
]


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------
@superkart_api.route('/', methods=['GET'])
def home():
    """
    Handles GET requests to the root URL.
    """
    logger.info("Home endpoint accessed")

    html = """
      <!DOCTYPE html>
      <html>
      <head>
        <title>SuperKart Sales API</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background-color: #f4f4f4;
          }
          h1 {
            color: #333;
            font-size: 3em;
          }
          p {
            color: #666;
            font-size: 1.5em;
            margin-top: 20px;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to SuperKart Sales Prediction API.</h1>
        <p>
          To obtain a single sales prediction, send a POST request to
          <code>/v1/predict</code>.
        </p>
        <p>
          For batch predictions, send a POST request to
          <code>/v1/predictbatch</code>.
        </p>
      </body>
      </html>
    """

    return html


# ---------------------------------------------------------
# Single inference endpoint
# ---------------------------------------------------------
@superkart_api.route('/v1/predict', methods=['POST'])
def predict_sales():
    """
    Handles single-row sales prediction.

    Expected JSON:
    {
        "Product_Weight": ...,
        "Product_Sugar_Content": "...",
        "Product_Allocated_Area": ...,
        "Product_MRP": ...,
        "Store_Size": "...",
        "Store_Location_City_Type": "...",
        "Store_Type": "...",
        "Store_Age_Years": ...,
        "Product_Type_Category": "...",
        "Product_Id_char": "..."
    }
    """

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Request body cannot be empty'
            }), 400

        # Check that all required fields are present
        missing_fields = [
            field for field in FEATURES
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'fields': missing_fields
            }), 400

        # Create DataFrame for one record
        input_data = pd.DataFrame(
            [[data[field] for field in FEATURES]],
            columns=FEATURES
        )

        logger.info(f"Single inference input:\n{input_data}")

        # Make prediction
        prediction = model.predict(input_data).tolist()[0]

        return jsonify({
            'Sales': prediction
        })

    except Exception as e:
        logger.exception("Single prediction failed")

        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


# ---------------------------------------------------------
# Batch inference endpoint
# ---------------------------------------------------------
@superkart_api.route('/v1/predictbatch', methods=['POST'])
def predict_sales_batch():
    """
    Handles batch sales prediction.

    Expected JSON:

    {
        "data": [
            {
                "Product_Weight": ...,
                "Product_Sugar_Content": "...",
                "Product_Allocated_Area": ...,
                "Product_MRP": ...,
                "Store_Size": "...",
                "Store_Location_City_Type": "...",
                "Store_Type": "...",
                "Store_Age_Years": ...,
                "Product_Type_Category": "...",
                "Product_Id_char": "..."
            },
            {
                "Product_Weight": ...,
                "Product_Sugar_Content": "...",
                "Product_Allocated_Area": ...,
                "Product_MRP": ...,
                "Store_Size": "...",
                "Store_Location_City_Type": "...",
                "Store_Type": "...",
                "Store_Age_Years": ...,
                "Product_Type_Category": "...",
                "Product_Id_char": "..."
            }
        ]
    }
    """

    try:
        request_data = request.get_json()

        if not request_data:
            return jsonify({
                'error': 'Request body cannot be empty'
            }), 400

        # Expect a list under the "data" key
        batch_data = request_data.get('data')

        if not isinstance(batch_data, list):
            return jsonify({
                'error': 'The "data" field must contain a list of records'
            }), 400

        if len(batch_data) == 0:
            return jsonify({
                'error': 'The batch cannot be empty'
            }), 400

        # Validate every record
        for index, record in enumerate(batch_data):

            if not isinstance(record, dict):
                return jsonify({
                    'error': f'Record at index {index} must be a JSON object'
                }), 400

            missing_fields = [
                field for field in FEATURES
                if field not in record
            ]

            if missing_fields:
                return jsonify({
                    'error': f'Missing required fields in record {index}',
                    'fields': missing_fields
                }), 400

        # Create DataFrame from all records
        input_data = pd.DataFrame(
            [
                [record[field] for field in FEATURES]
                for record in batch_data
            ],
            columns=FEATURES
        )

        logger.info(
            f"Batch inference requested for {len(input_data)} records"
        )

        # Make predictions for the entire batch
        predictions = model.predict(input_data).tolist()

        # Return predictions
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })

    except Exception as e:
        logger.exception("Batch prediction failed")

        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


# ---------------------------------------------------------
# Run Flask application
# ---------------------------------------------------------
if __name__ == '__main__':
    superkart_api.run(debug=True)
