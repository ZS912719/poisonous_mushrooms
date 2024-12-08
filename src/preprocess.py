import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class PreprocessDataset:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def __call__(self):
        processed = self.preprocess(self.df.copy())
        self.sample_train, self.sample_test = self.get_sample(processed)
        self.X_train, self.y_train = self.encode(self.sample_train)
        self.X_test, self.y_test = self.encode(self.sample_test)
        print("subset made and encoded")

    def __getitem__(self, idx):
        X_train = self.X_train[idx]
        y_train = self.y_train[idx]
        X_test = self.X_test[idx]
        y_test = self.y_test[idx]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def preprocess(df:pd.DataFrame):
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

    @staticmethod
    def get_sample(df:pd.DataFrame):
        """
        take out 500,000 samples randomly to do training
        """
        sample = df.sample(n=50, random_state= 0)
        df_train, df_test = train_test_split(sample, test_size=0.20, random_state=0)
        return df_train, df_test

    @staticmethod
    def encode(sample):
        """
        encode categorical columns with one-hot encoding.
        transform numerical columns with min-max scaling.
        :param sample:
        :return:
        """
        category = sample['class']
        features = sample.drop(["id","class"],axis=1)

        numerical_columns = features.select_dtypes(include=["int64", "float64"])
        categorical_columns = features.select_dtypes(exclude=["int64", "float64"])

        enc_onehot = preprocessing.OneHotEncoder()
        enc_minmax = preprocessing.MinMaxScaler()
        enc_label = preprocessing.LabelEncoder()

        onehot_array = enc_onehot.fit_transform(categorical_columns).toarray()
        minmax_array = enc_minmax.fit_transform(numerical_columns)
        label_array = enc_label.fit_transform(category)

        combined_input = np.concatenate([onehot_array, minmax_array], axis=1)

        return combined_input, label_array

