""" inference script """

import sys
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.joblib")

DEFAULT_SCORE_DATA = "data_1.json"

# Check if there are enough command line arguments
if len(sys.argv) > 1:
    DEFAULT_SCORE_DATA = sys.argv[1]

df = pd.read_json(DEFAULT_SCORE_DATA, orient="records")

# Make predictions for the new data
predictions = model.predict(df)

if predictions[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger did not Survive")
