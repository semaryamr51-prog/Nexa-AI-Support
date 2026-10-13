"""
model.py
--------
From-scratch embedding + LSTM/GRU intent classifier. Swap rnn_type between
"lstm" and "gru" to reproduce the one-line comparison suggested in the brief.
"""

import torch
import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_classes=4, pretrained_embeddings=None, n_layers=1,
                 rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn_type = rnn_type.lower()
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.rnn_type == "lstm":
            _, (hidden, _) = self.rnn(packed)
        else:
            _, hidden = self.rnn(packed)
        # concat final forward + backward hidden states
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(h))
