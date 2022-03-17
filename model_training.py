import logging
import os
import pickle
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from data_processing import Data, BoolFeatureBuilder, FeatureSelector

logging.getLogger().setLevel(logging.INFO)


class LeadScoringModel:
    def __init__(self, model, transform_steps=None):
        transform_steps = transform_steps or [('bool_feature_builder', BoolFeatureBuilder()),
                                              ('feature_selector', FeatureSelector())]

        self.transform_pipeline = Pipeline(steps=transform_steps)
        self.model = model
        self._name = 'lead_scoring_model'

    def fit(self, X, y, calibrate=False, cv=5):
        """ If calibrate is True will use cv for calibrating probabilities """
        logging.info(f'Fitting {self._name}')
        X_transformed = self.transform_pipeline.fit_transform(X.copy(), y)
        self.model.fit(X_transformed, y)

        if calibrate:
            logging.info(f'Calibrating {self._name}')
            self.model = CalibratedClassifierCV(base_estimator=self.model, cv=cv)
            self.model.fit(X_transformed, y)
        return

    def predict_proba(self, X):
        X_transformed = self.transform_pipeline.transform(X.copy())
        return self.model.predict_proba(X_transformed)


class ModelSaver:
    def __init__(self):
        self.model_dir = r'C:\Users\Marin\Documents\trellis\code\lead_conversion_model'
        self.date_stamp = str(dt.date.today())

    def save(self, model):
        file_path = os.path.join(self.model_dir, f'{model._name}_{self.date_stamp}.pkl')
        with open(file_path, 'wb') as output_file:
            pickle.dump(model, output_file)
        logging.info(f'Saved {model._name} at {file_path}')

    def load(self, file_name):
        file_path = os.path.join(self.model_dir, f'{file_name}.pkl')
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model


def main():
    data = Data()
    data.read_data()

    model = LeadScoringModel(model=RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features='auto',
                                                          n_jobs=-1))
    model.fit(data.X, data.y, calibrate=True)  # use full dataset for final model

    model_saver = ModelSaver()
    model_saver.save(model)


if __name__ == '__main__':
    main()
