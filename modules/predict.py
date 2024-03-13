# <YOUR_IMPORTS>
import os
import dill
import pandas as pd
import json
import logging


path = os.environ.get('PROJECT_PATH', '')
def loading_model():
    models_dir = f"{path}/data/models/"

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if model_files:
        latest_model_file = \
        sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)[0]

        model_path = os.path.join(models_dir, latest_model_file)
        logging.info(model_path)
        with open(model_path, "rb") as f:
            model = dill.load(f)

        return model

def save_predictions(predictions):

    predictions_df = pd.DataFrame({'predictions': predictions})
    predictions_df.to_csv(f"{path}/data/predictions/test.csv", index=False)

def predict():
    # <YOUR_CODE>
    model = loading_model()
    prediction_results = []



    directory = f"{path}/data/test/"
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            test_data = pd.DataFrame(data, index=[0])
            predictions = model.predict(test_data)
            prediction_results.extend(predictions)

    save_predictions(prediction_results)



if __name__ == "__main__":
    predict()