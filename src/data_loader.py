import pandas as pd

TRAIN_DATA_PATH = "../data/raw/train.csv"
TEST_DATA_PATH = "../data/raw/test.csv"

df_train = pd.read_csv(TRAIN_DATA_PATH)
df_test = pd.read_csv(TEST_DATA_PATH)

