from logging import getLogger
from typing import Dict, Any, Optional

import torch
from torch import nn
from transformers import AutoModel


logger = getLogger(__name__)


class MTBModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "bert-base-cased",
        variant: str = "a",
        vocab_size: int = 29000,
        num_classes: int = 42,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = variant
        self.vocab_size = vocab_size

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.encoder.resize_token_embeddings(self.vocab_size)

        self.hidden_size = self.encoder.config.hidden_size
        self.in_features = (
            self.hidden_size if self.variant in ["a", "d"] else 2 * self.hidden_size
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.in_features, out_features=num_classes),
        )

    def forward(
        self, x: Dict[str, Any], cues: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # shape: [batch_size, max_seq_len_per_batch, hidden_size]
        # (in this case, hidden_size is 768 for bert)
        out = self.encoder(**x).last_hidden_state

        if self.variant in ["a", "d"]:
            out = self.fetch_feature_a_or_d(out)
        elif self.variant in ["b", "e"]:
            out = self.fetch_feature_b_or_e(out)
        elif self.variant == "f":
            out = self.fetch_feature_f(out)

        logger.warning(out.shape)
        out = self.fc(out)
        return out

    def fetch_feature_a_or_d(self, embedding: torch.Tensor) -> torch.Tensor:
        """Fetch feature for variant 'a' or 'd', i.e. gather the embeddings
        at [CLS] positions.

        Args:
            embedding: The text embedding of shape `[batch_size, max_seq_len_per_batch,
            hidden_size]`.

        Returns:
            The [CLS] embedding, of shape `[batch_size, hidden_size]
        """
        return torch.squeeze(embedding[:, 0, :], dim=1)

    def fetch_feature_b_or_e(
        self, embedding: torch.Tensor, cues: torch.Tensor
    ) -> torch.Tensor:
        """Fetch feature for variant 'a' or 'd', i.e. gather the embeddings at
        [CLS] positions.

        Args:
            embedding: The text embedding of shape `[batch_size, max_seq_len_per_batch,
            hidden_size]`.

        Returns:

        """

        return

    def fetch_feature_f(
        self, embedding: torch.Tensor, cues: torch.Tensor
    ) -> torch.Tensor:
        """Fetch feature for variant 'f', i.e. gather the embeddings at entity-start cues.
        The `cues` are of shape `[2, batch_size]`.
        """
        start_e1, start_e2 = cues
        cues = cues.expand((cues.shape[0], 2, self.hidden_size))

        out = torch.gather(embedding, 1, cues)
        out = torch.reshape(out, (out.shape[0], -1))
        out = self.fc(out)
        return
