import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.binaryClassificationModel import BinaryClassificationModel


class Prediction:
    def __init__(self, df, model_path):
        self.df = df
        self.model_path = model_path

    def load_model(self):
        model = BinaryClassificationModel(self.df.shape[1])
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()
        return model

    @staticmethod
    def predict(model, dataloader, device):
        predictions = []
        ids = []
        with torch.no_grad():
            for features, sample_ids in dataloader:
                features = features.to(device)
                outputs = model(features)
                preds = torch.sigmoid(outputs).round()
                predictions.extend(preds.cpu().numpy().flatten())
                ids.extend(sample_ids)
        return ids, predictions
