"""
Feedforward neural network for fraud classification, using the same
engineered tabular feature set as the XGBoost/Random Forest baselines
(src/preprocessing.py) -- so its results are directly comparable to
Milestone 3's, not confounded by a different feature set.
"""

import torch
import torch.nn as nn


class FraudFNN(nn.Module):
    """
    A straightforward MLP: Linear -> BatchNorm -> ReLU -> Dropout, repeated
    for each hidden layer, ending in a single output logit (use
    BCEWithLogitsLoss / torch.sigmoid on the output, not a Sigmoid layer
    baked in -- BCEWithLogitsLoss is numerically more stable).
    """

    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # (batch, 1) -> (batch,) logits


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, device, batch_size: int = 8192):
    """
    Batched inference -> sigmoid probabilities, as a 1-D numpy array.
    Batched (rather than one giant forward pass) so this scales to however
    large a split gets without a memory spike, and works the same whether
    called on val, test, or a hybrid-framework stacking set.
    """
    model.eval()
    probs = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size].to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()
