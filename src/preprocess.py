import pandas as pd
from sklearn import preprocessing


class DataSet:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def __call__(self):
        processed = self.preprocess(self.df.copy())
        self.sample = self.get_sample(processed)
        self.X_cat, self.X_num, self.y = self.encode(self.sample)
        print("subset made and encoded")

    def __getitem__(self, idx):
        X_cat = self.X_cat[idx]
        X_num = self.X_num[idx]
        y = self.y[idx]
        return X_cat, X_num, y

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
        return df.sample(n=50, random_state= 0)

    @staticmethod
    def encode(sample):
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

        return onehot_array, minmax_array, label_array

