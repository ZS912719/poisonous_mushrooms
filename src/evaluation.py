import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.binaryClassificationModel import BinaryClassificationModel

class Evaluation:
    def __init__(self,X,y,model_path):
        self.X = X
        self.y = y
        self.model_path = model_path

    def __call__(self, *args, **kwargs):
        model = BinaryClassificationModel(self.X.shape[1])
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()

        print("Model loaded successfully.")

        eval_data = list(zip(self.X, self.y))
        eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False)

        metrics = self.evaluate_model(model, eval_loader)
        print("Evaluation Metrics:", metrics)

    @staticmethod
    def evaluate_model(model, dataloader):
        """
        Evaluate the performance of a pre-trained Binary Classification Model.

        Args:
            model (BinaryClassificationModel): The PyTorch model to evaluate.
            dataloader (DataLoader): DataLoader for the evaluation dataset.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in dataloader:
                outputs = model(features)
                preds = torch.sigmoid(outputs).round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }


