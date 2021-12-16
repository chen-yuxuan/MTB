from logging import getLogger
from typing import List

import torch
from torch import nn
from transformers import AutoModel


logger = getLogger(__name__)


class MTBModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-cased",
        hidden_size: int = 256,
        num_classes: int = 42,
        dropout: float = 0.1,
    ):

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.Dropout(p=dropout),
        )

    def forward(
        self, x: torch.Tensor, variant: str = "f", positions: List[int] = None
    ) -> torch.Tensor:
        return x
