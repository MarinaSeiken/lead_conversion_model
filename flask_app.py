import pickle
import pandas as pd
from flask import Flask, request, jsonify

from model_training import LeadScoringModel


app = Flask(__name__)
model = pickle.load(open('model/lead_scoring_model_2022-03-19.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)
    data = pd.DataFrame.from_records([data])
    app.logger.info(f'{data.shape}')

    prediction = model.predict_proba(data)[:,1]
    app.logger.info(prediction)

    return jsonify(prediction[0])


if __name__ == "__main__":
    app.run()
