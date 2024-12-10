import pandas as pd

TRAIN_DATA_PATH = "../data/raw/train.csv"
TEST_DATA_PATH = "../data/raw/test.csv"

train_set = pd.read_csv(TRAIN_DATA_PATH)
test_set = pd.read_csv(TEST_DATA_PATH)

