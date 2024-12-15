import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.binaryClassificationModel import BinaryClassificationModel

def train_one_epoch(model, dataloader, device, lr):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for X_batch, y_batch in dataloader:
        y_batch = y_batch.unsqueeze(1).to(device)
        X_batch = X_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = model.loss(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

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


