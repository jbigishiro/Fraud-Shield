"""
LSTM for fraud classification over per-card transaction sequences.

Architecture: an LSTM consumes the sequence of dynamic features (amount,
velocity, z-score, time-of-day -- see sequence_features.SEQ_FEATURES), and
its final hidden state is concatenated with static context about the
current transaction (customer age, city population, category, gender) --
the same split used in train_fnn.py/train_lstm.py: things that vary
transaction-to-transaction go through the sequence, things that describe
the customer/merchant context go in once.
"""

import torch
import torch.nn as nn


class FraudLSTM(nn.Module):
    def __init__(
        self,
        seq_input_dim: int,
        static_input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_static):
        # x_seq: (batch, seq_len, seq_input_dim)
        _, (h_n, _) = self.lstm(x_seq)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's hidden state
        combined = torch.cat([last_hidden, x_static], dim=1)
        return self.head(combined).squeeze(-1)  # logits


@torch.no_grad()
def predict_proba(model, X_seq: torch.Tensor, X_static: torch.Tensor, device, batch_size: int = 4096):
    """Batched inference -> sigmoid probabilities, as a 1-D numpy array."""
    model.eval()
    probs = []
    for start in range(0, len(X_seq), batch_size):
        seq_batch = X_seq[start:start + batch_size].to(device)
        static_batch = X_static[start:start + batch_size].to(device)
        logits = model(seq_batch, static_batch)
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()
