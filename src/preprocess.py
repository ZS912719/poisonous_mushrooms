import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataSet:

    def __init__(self, df:pd.DataFrame, seed:int):
        self.df = df
        self.rng = np.random.default_rng(seed)
        self.df_processed = None
        self.df_sample = None

    def __len__(self):
        return len(self.df_sample)

    def preprocess(self):
        """
        Preprocess the dataset:
        1. Remove columns with more than 60% missing values.
        2. For columns with lesser missing values, replace NaN:
            - With mode for categorical columns.
            - With median for numerical columns.
        """
        df = self.df.copy()

        threshold = 0.6

        high_missing_cols = df.columns[df.isnull().mean() > threshold]
        df_processed = df.drop(columns=high_missing_cols)

        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if df_processed[col].dtype == 'object':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        self.df_processed = df_processed

        return self

    def get_sample(self):
        """
        take out 500,000 samples randomly to do training
        """
        df = self.df_processed
        df_sample = df.sample(n=500000, random_state=self.rng)
        self.df_sample = df_sample

    def split(self):
        sample = self.df_sample.copy().drop("class", axis=1)
        numerical_columns = sample.select_dtypes(include=["int64", "float64"]).columns
        categorical_columns = sample.select_dtypes(exclude=["int64", "float64"]).columns
        return numerical_columns, categorical_columns

    def __call__(self):
        self.preprocess()
        self.get_sample()
        print("subset made and encoded")

    def __getitem__(self, idx):

        sample = self.df_sample.iloc[idx]
        part_X = sample.drop("class")
        part_y = sample["class"]

        numerical_columns, categorical_columns = self.split()

        X_num = []
        if len(numerical_columns) > 0:
            scaler = MinMaxScaler()
            scaler.fit(self.df_sample[numerical_columns])
            X_num = scaler.transform(part_X[numerical_columns].values.reshape(1, -1)).flatten()

        X_cat = []
        for col in categorical_columns:
            unique_values = self.df_sample[col].unique()
            unique_map = {v: i for i, v in enumerate(unique_values)}
            onehot_array = np.zeros(len(unique_values), float)
            onehot_index = unique_map[part_X[col]]
            onehot_array[onehot_index] = 1
            X_cat.append(onehot_array)

        X = np.concatenate([X_num] + X_cat)

        y = 0 if part_y == 'e' else 1

        return X, y