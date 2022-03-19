import logging
import os
import pickle
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.data_processing import Data, BoolFeatureBuilder, FeatureSelector


logging.getLogger().setLevel(logging.INFO)


class LeadScoringModel:
    def __init__(self, model, transform_steps):
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
        folder_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_dir = os.path.join(folder_dir, '../model')
        self.date_stamp = str(dt.date.today())

    def save(self, model):
        file_path = os.path.join(self.model_dir, f'{model._name}_{self.date_stamp}.pkl')
        with open(file_path, 'wb') as output_file:
            pickle.dump(model, output_file)
        logging.info(f'Saved {model._name} at {file_path}')


def main():
    data = Data()
    data.read_data()

    cat_col_dict = {
        'A free copy of Mastering The Interview': ['Yes'],
        'City': ['Other Cities', 'Thane & Outskirts'],
        'Do Not Email': ['Yes'],
        'Last Activity': ['Email Opened', 'Olark Chat Conversation', 'Page Visited on Website', 'SMS Sent'],
        'Last Notable Activity': ['Email Opened', 'Modified', 'SMS Sent'],
        'Lead Origin': ['Landing Page Submission', 'Lead Add Form'],
        'Lead Quality': ['Low in Relevance', 'Might be', 'Not Sure', 'Worst'],
        'Lead Source': ['Direct Traffic', 'Google', 'Olark Chat', 'Organic Search'],
        'Specialization': ['Finance Management', 'Human Resource Management', 'Marketing Management',
                           'Operations Management', 'Others'],
        'Tags': ['Interested in other courses', 'Ringing', 'Will revert after reading the email'],
        'What is your current occupation': ['Unemployed', 'Working Professional']
    }

    model_cols = [
        'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Lead Origin_Landing Page Submission',
        'Lead Origin_Lead Add Form', 'Lead Source_Direct Traffic', 'Lead Source_Google', 'Lead Source_Olark Chat',
        'Lead Source_Organic Search', 'Do Not Email_Yes', 'Last Activity_Email Opened',
        'Last Activity_Olark Chat Conversation', 'Last Activity_Page Visited on Website', 'Last Activity_SMS Sent',
        'Specialization_Finance Management', 'Specialization_Human Resource Management',
        'Specialization_Marketing Management', 'Specialization_Operations Management', 'Specialization_Others',
        'What is your current occupation_Unemployed', 'What is your current occupation_Working Professional',
        'Tags_Interested in other courses', 'Tags_Ringing', 'Tags_Will revert after reading the email',
        'Lead Quality_Low in Relevance', 'Lead Quality_Might be', 'Lead Quality_Not Sure', 'Lead Quality_Worst',
        'City_Other Cities', 'City_Thane & Outskirts', 'A free copy of Mastering The Interview_Yes',
        'Last Notable Activity_Email Opened', 'Last Notable Activity_Modified', 'Last Notable Activity_SMS Sent']

    # use full dataset for final model
    model = LeadScoringModel(
        transform_steps=[('bool_feature_builder', BoolFeatureBuilder(cat_col_dict)),
                         ('feature_selector', FeatureSelector(model_cols))],
        model=RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features='auto', n_jobs=-1))
    model.fit(data.X, data.y, calibrate=True)

    model_saver = ModelSaver()
    model_saver.save(model)


if __name__ == '__main__':
    main()
