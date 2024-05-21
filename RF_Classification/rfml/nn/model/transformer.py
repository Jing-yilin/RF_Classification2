"""Simplistic convolutional neural network.
"""

__author__ = "Yilin Jing <yj220@sussex.ac.uk>"

# External Includes
import torch.nn as nn

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Callable, Any, List
from .base import Model


def seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Transformer(Model):
    def __init__(
        self,
        input_samples: int = 128,
        n_classes: int = 11,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(input_samples, n_classes)
        self.num_heads = num_heads
        self.num_layers = num_layers

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Embedding layer: change the input feature dimension to the appropriate size
        self.embedding = nn.Linear(2, input_samples)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, input_samples, input_samples)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_samples, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Fully connected layers
        self.fc1 = nn.Linear(input_samples * input_samples, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Assuming x has shape (batch_size, channels, feature_dim, sequence_length)
        # x.shape = torch.Size([256, 1, 2, 128])
        batch_size, channels, feature_dim, sequence_length = x.shape

        # Reshape x to (batch_size * channels, sequence_length, feature_dim)
        x = x.view(batch_size * channels, sequence_length, feature_dim)

        # Add positional encoding
        # Linear layer expects input of shape (batch_size * channels, sequence_length, input_samples)
        x = self.embedding(x)
        x += self.positional_encoding[:, :sequence_length, :]

        # Transformer Encoder
        # Transformer expects input shape (sequence_length, batch_size * channels, input_samples)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # Convert back to shape (batch_size * channels, sequence_length, input_samples)
        x = x.permute(1, 0, 2)

        # Flatten and fully connected layers
        x = x.contiguous().view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def transformer_model(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> Transformer:
    model = Transformer(
        input_samples=128, n_classes=11, num_heads=8, num_layers=6, **kwargs
    )
    return model
