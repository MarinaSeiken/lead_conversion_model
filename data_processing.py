import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class Data:
    def __init__(self):
        self.target_name = 'Converted'
        self.X = None
        self.y = None

    def read_data(self):
        file_path = r'C:\Users\Marin\Documents\trellis\code\lead_conversion_model\data\leads.csv'
        df = pd.read_csv(file_path, index_col='Lead Number')

        self.X = df.drop(self.target_name, axis=1)
        self.y = df[self.target_name]
        logging.info(f'Loaded data from {file_path}')

    def split(self, with_validation=False):
        if with_validation:
            return self.split_into_three_sets()
        else:
            return self.split_into_two_sets()

    def split_into_three_sets(self, test_size=0.25, valid_size=0.25):
        test_and_valid_size = test_size + valid_size
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_and_valid_size,
                                                            random_state=42)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=test_size/test_and_valid_size,
                                                            random_state=42)
        return X_train, X_test, X_valid, y_train, y_test, y_valid

    def split_into_two_sets(self, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test


class BoolFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, input_cols_dict=None):
        self.input_cols_dict = input_cols_dict or {
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
            'What is your current occupation': ['Unemployed', 'Working Professional']}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_cols = list(set(self.input_cols_dict.keys()).difference(set(X.columns)))

        assert len(missing_cols) == 0, f"Missing columns: {', '.join(missing_cols)}"
        for col, values in self.input_cols_dict.items():
            for val in values:
                X[f'{col}_{val}'] = X[col] == val
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols):
        self.log_cols = log_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_cols = list(set(self.log_cols).difference(set(X.columns)))
        assert len(missing_cols) == 0, f"Missing columns: {', '.join(missing_cols)}"

        for col in self.log_cols:
            if X[col].min() >= 0:
                print(f'Transforming {col}')
                X[col] = np.log(X[col] + 1)
            else:
                logging.warning(f'Min value has to be >= 0 for log transform, got {X[col].min()} for {col}')
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model_cols=None):
        self.model_cols = model_cols or [
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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_cols = list(set(self.model_cols).difference(set(X.columns)))
        assert len(missing_cols) == 0, f"Missing columns: {', '.join(missing_cols)}"

        return X[self.model_cols]

