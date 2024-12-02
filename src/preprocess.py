import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataSet:

    def __init__(self, df:pd.DataFrame, seed:int):
        self.df = df
        self.rng = np.random.default_rng(seed)
        self.df_processed = None
        self.df_sample = None
        self.df_encoded = None

    def __len__(self):
        return len(self.df)

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

    def onehot(self):
        """
        Normalize the numeric columns with min_max_scaler.
        Normalize the categorical columns with one-hot encoding.
        """
        df = self.df_processed

        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        self.df_encoded= df_encoded

    def __call__(self):
        self.preprocess()
        self.get_sample()
        self.onehot()
        print("subset made and encoded")

    # def __getitem__(self, idx):

