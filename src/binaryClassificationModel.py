import torch
import torch.nn as nn

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        """
        Make predictions for binary classification.

        Args:
            x: Input features (torch.Tensor).

        Returns:
            Binary predictions (torch.Tensor): A tensor of 0s and 1s.
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        return predictions

    @staticmethod
    def loss(x, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(x, y)