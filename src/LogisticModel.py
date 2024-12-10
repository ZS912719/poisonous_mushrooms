import math

import numpy as np
from scipy.special import softmax


class LogisticModel:
    weights: np.ndarray

    def __init__(self, input_dim: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        lim = 1 / math.sqrt(input_dim)
        self.weights = rng.uniform(-lim, lim, (n_classes, input_dim + 1))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Do the forward pass.

        ---
        Args:
            X: A batch of images.
                Shape of [batch_size, input_dim].

        ---
        Returns:
            The predictions.
                Shape of [batch_size, n_classes].
        """
        X = np.pad(X, [(0, 0), (0, 1)], constant_values=1)
        logits = X @ self.weights.T
        return softmax(logits, axis=1)

    def update(self, grad: np.ndarray, lr: float):
        """Update the weights based on the given gradient and learning rate.

        ---
        Args:
            grad: The gradient.
                Shape of `self.weights`.
            lr: The learning rate of the update.
        """
        self.weights = self.weights - lr * grad

    @staticmethod
    def gradient(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the gradient.

        ---
        Args:
            X: Batch of images.
                Shape of [batch_size, input_dim].
            y_pred: Model's predictions.
                Shape of [batch_size, n_classes].
            y_true: Ground truth.
                Shape of [batch_size, n_classes].

        ---
        Returns:
            The gradient.
                Shape of `self.weights`.
        """
        X = np.pad(X, [(0, 0), (0, 1)], constant_values=1)
        return -(y_true - y_pred).T @ X / len(X)