import pandas as pd


class Preprocess:

    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.processed = self.preprocess(self.df.copy())

    @staticmethod
    def preprocess(df:pd.DataFrame)->pd.DataFrame:
        """
        Preprocess the df_train:
        1. Remove columns with more than 60% missing values.
        2. For columns with lesser missing values, replace NaN:
            - With mode for categorical columns.
            - With median for numerical columns.
        """
        threshold = 0.6

        high_missing_cols = df.columns[df.isnull().mean() > threshold]
        processed = df.drop(columns=high_missing_cols)

        for col in processed.columns:
            if processed[col].isnull().any():
                if processed[col].dtype == 'object':
                    processed[col] = processed[col].fillna(processed[col].mode()[0])
                else:
                    processed[col] = processed[col].fillna(processed[col].median())

        return processed





