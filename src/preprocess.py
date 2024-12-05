import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataSet:

    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.df_sample = None
        self.X = None

    @staticmethod
    def preprocess(df:pd.DataFrame):
        """
        Preprocess the dataset:
        1. Remove columns with more than 60% missing values.
        2. For columns with lesser missing values, replace NaN:
            - With mode for categorical columns.
            - With median for numerical columns.
        """
        threshold = 0.6

        high_missing_cols = df.columns[df.isnull().mean() > threshold]
        df_processed = df.drop(columns=high_missing_cols)

        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if df_processed[col].dtype == 'object':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        return df_processed

    @staticmethod
    def get_sample(df:pd.DataFrame):
        """
        take out 500,000 samples randomly to do training
        """
        df_sample = df.sample(n=50, random_state= 0)
        return df_sample

    @staticmethod
    def split_features(sample):
        sample = sample.drop(["id","class"],axis=1)
        numerical_columns = sample.select_dtypes(include=["int64", "float64"])
        categorical_columns = sample.select_dtypes(exclude=["int64", "float64"])
        return numerical_columns, categorical_columns
    
    def encode(self):
        sample = self.df_sample
        # y = sample["class"]
        numerical_columns, categorical_columns = self.split_features(sample)
        enc_onehot = preprocessing.OneHotEncoder()

        # X_num = ['test']
        X_cat = []
        
        enc_onehot.fit(categorical_columns)
        onehot_array = enc_onehot.transform(categorical_columns).toarray()
        X_cat.append(onehot_array)
        self.X =  X_cat


    def __call__(self):
        df_processed = self.preprocess(self.df.copy())
        self.df_sample = self.get_sample(df_processed)
        self.encode()
        print("subset made and encoded")

    
    def __getitem__(self, idx):
        X = self.X[idx]
        return X