import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing


class ProcessedDataSet:
    def __init__(self, sample: pd.DataFrame, feature_order=None):
        self.sample = sample
        self.feature_order = feature_order
        self.enc_onehot = None
        self.enc_minmax = None

    def __call__(self):
        self.X, self.y = self.encode(self.sample)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.safe_index(self.y, idx, default=-1)
        return X, y

    def __len__(self):
        return self.X.shape[0]

    def encode(self, sample):
        """
        encode categorical columns with one-hot encoding.
        transform numerical columns with min-max scaling.
        :param sample:
        :return:
        """
        label_array = []
        if 'class' in sample.columns:
            category = sample['class']
            features = sample.drop(["id", "class"], axis=1)

            enc_label = preprocessing.LabelEncoder()
            label_array = enc_label.fit_transform(category)
        else:
            features = sample.drop(["id"], axis=1)

        numerical_columns = features.select_dtypes(include=["int64", "float64"])
        categorical_columns = features.select_dtypes(exclude=["int64", "float64"])

        if self.feature_order is None:
            self.enc_onehot = preprocessing.OneHotEncoder(handle_unknown='ignore')
            self.enc_minmax = preprocessing.MinMaxScaler()

            onehot_array = self.enc_onehot.fit_transform(categorical_columns).toarray()
            minmax_array = self.enc_minmax.fit_transform(numerical_columns)

            self.feature_order = list(categorical_columns.columns) + list(numerical_columns.columns)
        else:
            categorical_columns = categorical_columns.reindex(
                columns=self.feature_order[:len(categorical_columns.columns)], fill_value=0)
            numerical_columns = numerical_columns.reindex(columns=self.feature_order[len(categorical_columns.columns):],
                                                          fill_value=0)

            onehot_array = self.enc_onehot.transform(categorical_columns).toarray()
            minmax_array = self.enc_minmax.transform(numerical_columns)

        feature_array = np.concatenate([onehot_array, minmax_array], axis=1)

        return (
            torch.tensor(feature_array, dtype=torch.float32),
            torch.tensor(label_array, dtype=torch.float32)
        )

    @staticmethod
    def safe_index(data, idx, default=None):
        if 0 <= idx < len(data):
            return data[idx]
        else:
            return default