""" model online endpoint scoring script"""

import os
import json
import pandas as pd
import joblib


def init():
    """
    This function is called when the container is initialized/started
    """
    global model

    model_path = os.path.join(os.environ.get("AZUREML_MODEL", "model.joblib"))
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


def run(raw_data):
    """
    This function is called for every invocation of the endpoint
    to perform the actual scoring/prediction.
    """

    json_data = json.loads(raw_data)
    data = json_data[0]

    try:
        features = [
            "PassengerId",
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_female",
            "Sex_male",
            "Embarked_C",
            "Embarked_Q",
            "Embarked_S",
        ]

        prediction_data = [data[feature] for feature in features]
        prediction_df = pd.DataFrame([prediction_data], columns=features)
        print("prediction_data: %s", prediction_data)

        result = model.predict(prediction_df)
        print("Result: %s", result)

        # create a JSON response object
        prediction = {"prediction": int(result[0])}
        return prediction

    except ValueError as err:
        print(str(err))
        return str(err)
