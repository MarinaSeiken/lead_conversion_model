import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class Data:
    def __init__(self):
        self.target_name = 'Converted'
        self.X = None
        self.y = None

    def read_data(self):
        folder_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(folder_dir, '../data/leads.csv')
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
    def __init__(self, input_cols_dict):
        self.input_cols_dict = input_cols_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_cols = list(set(self.input_cols_dict.keys()).difference(set(X.columns)))

        assert len(missing_cols) == 0, f"Missing columns: {', '.join(missing_cols)}"
        for col, values in self.input_cols_dict.items():
            for val in values:
                X[f'{col}_{val}'] = X[col] == val
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_cols = list(set(self.features).difference(set(X.columns)))
        assert len(missing_cols) == 0, f"Missing columns: {', '.join(missing_cols)}"

        return X[self.features]

