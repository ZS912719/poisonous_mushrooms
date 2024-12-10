import pandas as pd
import torch

from torch.utils.data import Dataset


class BinaryClassificationDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: List of tuples, where each tuple is (features, label).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
